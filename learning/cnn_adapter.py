"""PyTorch execution for framework-neutral CNN specs and traceable arithmetic."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .cnn_spec import CNNModelSpec, pair, output_axis


class CNNAdapter:
    framework = 'pytorch'

    def build(self, spec: CNNModelSpec):
        if spec.framework != self.framework:
            raise NotImplementedError('Only the PyTorch CNN adapter is implemented in M5')
        shapes=spec.infer_shapes()
        modules=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            for row,layer in zip(shapes[1:], spec.layers):
                previous=shapes[row['index']-1]['shape']
                if layer.kind == 'conv2d':
                    modules.append(nn.Conv2d(previous[1],layer.out_channels,layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,dilation=layer.dilation,bias=layer.bias))
                elif layer.kind == 'relu': modules.append(nn.ReLU())
                elif layer.kind == 'maxpool2d': modules.append(nn.MaxPool2d(layer.kernel_size,layer.stride))
                elif layer.kind == 'avgpool2d': modules.append(nn.AvgPool2d(layer.kernel_size,layer.stride))
                elif layer.kind == 'flatten': modules.append(nn.Flatten())
                elif layer.kind == 'linear': modules.append(nn.Linear(previous[1],layer.out_features,bias=layer.bias))
        return nn.Sequential(*modules)

    def trace(self,spec,x=None):
        model=self.build(spec).eval()
        if x is None:
            x=torch.arange(spec.input_channels*spec.image_height*spec.image_width,dtype=torch.float32).reshape(1,spec.input_channels,spec.image_height,spec.image_width)
            x=x/max(1.,x.max().item())
        expected=(1,spec.input_channels,spec.image_height,spec.image_width)
        if tuple(x.shape)!=expected: raise ValueError(f'Expected input {expected}')
        steps=[{'operator':'Input','shape':tuple(x.shape),'tensor':x.detach().clone()}]
        with torch.no_grad():
            for layer in model:
                x=layer(x)
                steps.append({'operator':type(layer).__name__,'shape':tuple(x.shape),'tensor':x.clone()})
        return model,steps


def filter_presets(name,channels=1):
    templates={
        'Identity': [[0,0,0],[0,1,0],[0,0,0]],
        'Vertical edge': [[-1,0,1],[-1,0,1],[-1,0,1]],
        'Horizontal edge': [[-1,-1,-1],[0,0,0],[1,1,1]],
        'Blur': [[1/9]*3 for _ in range(3)],
        'Sharpen': [[0,-1,0],[-1,5,-1],[0,-1,0]],
    }
    if name not in templates: raise ValueError('Unknown filter preset')
    return torch.tensor(templates[name],dtype=torch.float32).repeat(channels,1,1)/channels


def convolution_explorer(input_tensor,weights,bias=None,stride=1,padding=0,activation='None',row=0,col=0,filter_index=0):
    """Read selected patch, per-channel products and actual torch Conv2d output.

    Tensors: input [C,H,W], weight [O,C,Kh,Kw], optional bias [O].
    PyTorch performs cross-correlation (not a flipped-kernel convolution).
    """
    if input_tensor.ndim != 3 or weights.ndim != 4 or input_tensor.shape[0] != weights.shape[1]:
        raise ValueError('Expected input CHW and weights OIHW with matching channels')
    if not torch.isfinite(input_tensor).all() or not torch.isfinite(weights).all():
        raise ValueError('Nonfinite input/weights')
    if bias is not None and (bias.shape != (weights.shape[0],) or not torch.isfinite(bias).all()):
        raise ValueError('Invalid bias vector')
    sh,sw=pair(stride);ph,pw=pair(padding)
    if min(sh,sw)<1 or min(ph,pw)<0: raise ValueError('Invalid stride/padding')
    y=F.conv2d(input_tensor[None],weights,bias,stride=(sh,sw),padding=(ph,pw))[0]
    if not (0 <= filter_index < y.shape[0] and 0 <= row < y.shape[1] and 0 <= col < y.shape[2]):
        raise ValueError('Convolution position/filter out of bounds')
    padded=F.pad(input_tensor,(pw,pw,ph,ph))
    kh,kw=weights.shape[-2:];patch=padded[:,row*sh:row*sh+kh,col*sw:col*sw+kw]
    products=patch*weights[filter_index]
    sums=products.flatten(1).sum(1)
    linear=y[filter_index,row,col]
    activated=F.relu(y) if activation == 'ReLU' else y
    if activation not in ('None','ReLU'): raise ValueError('Activation must be None/ReLU')
    return {'output':y,'activated':activated,'patch':patch,'products':products,
            'channel_contributions':sums,'sum_products':sums.sum().item(),
            'bias':0.0 if bias is None else bias[filter_index].item(),
            'linear_value':linear.item(),'activated_value':activated[filter_index,row,col].item(),
            'window_origin':(row*sh-ph,col*sw-pw)}


def pooling_explorer(feature_map,kind='max',kernel_size=2,stride=None,row=0,col=0):
    if feature_map.ndim != 2: raise ValueError('Expected 2D feature map')
    kh,kw=pair(kernel_size);sh,sw=pair(stride if stride is not None else kernel_size)
    if kind not in ('max','average'): raise ValueError('Invalid pooling kind')
    if min(kh,kw,sh,sw)<1: raise ValueError('Invalid pooling dimensions')
    t=feature_map[None,None]
    y=(F.max_pool2d(t,(kh,kw),(sh,sw)) if kind=='max'
       else F.avg_pool2d(t,(kh,kw),(sh,sw)))[0,0]
    if not (0<=row<y.shape[0] and 0<=col<y.shape[1]): raise ValueError('Pooling position out of bounds')
    window=feature_map[row*sh:row*sh+kh,col*sw:col*sw+kw]
    return {'output':y,'window':window,'value':y[row,col].item(),'origin':(row*sh,col*sw)}


def single_sgd_step(spec: CNNModelSpec, x, targets, lr=.01):
    if lr<=0 or not math.isfinite(lr): raise ValueError('Learning rate must be positive and finite')
    model=CNNAdapter().build(spec)
    opt=torch.optim.SGD(model.parameters(),lr=lr)
    opt.zero_grad(set_to_none=True)
    logits=model(x)
    loss=F.cross_entropy(logits,targets)
    loss.backward()
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    grads={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    opt.step()
    after={name:p.detach().clone() for name,p in model.named_parameters()}
    with torch.no_grad(): after_loss=F.cross_entropy(model(x),targets).item()
    return {'logits':logits.detach(),'loss_before':loss.item(),'loss_after':after_loss,
            'parameters':{name:{'before':before[name],'gradient':grads[name],
                                'after':after[name],'gradient_norm':grads[name].norm().item()}
                          for name in before}}
