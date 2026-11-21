"""LSTM execution, gate arithmetic and autograd; no presentation dependencies."""
import math
import torch
from torch import nn
from .rnn_adapter import RecurrentModel
from .lstm_spec import LSTMModelSpec

GATES = ('input', 'forget', 'candidate', 'output')  # PyTorch packed order i,f,g,o


class LSTMModel(RecurrentModel):
    def __init__(self, spec):
        nn.Module.__init__(self)
        self.spec = spec
        self.rnn = nn.LSTM(spec.input_size, spec.hidden_size, spec.num_layers,
                           bias=spec.bias, batch_first=spec.batch_first, dropout=spec.dropout)
        self.head = nn.Linear(spec.hidden_size, spec.output_size, bias=spec.bias)

    def forward(self, x, lengths=None):
        output, (hidden, _) = self.encode(x, lengths)
        return self.head(hidden[-1] if self.spec.selection == 'final' else output)


class LSTMAdapter:
    def build(self, spec: LSTMModelSpec):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            return LSTMModel(spec)

    def trace(self, spec, x):
        model = self.build(spec).eval()
        rows = [{'name':'Sequence input', 'tensor':x.detach()}]
        values = x
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for layer in range(spec.num_layers):
                cell = nn.LSTM(spec.input_size if layer == 0 else spec.hidden_size,
                               spec.hidden_size, bias=spec.bias, batch_first=spec.batch_first)
                cell.load_state_dict({k.replace(f'_l{layer}', '_l0'):v
                    for k,v in model.rnn.state_dict().items() if k.endswith(f'_l{layer}')})
                values, (hidden, memory) = cell(values)
                rows.extend([{'name':f'Layer {layer+1} output', 'tensor':values},
                             {'name':f'Layer {layer+1} h_n', 'tensor':hidden},
                             {'name':f'Layer {layer+1} c_n', 'tensor':memory}])
            selected = hidden[-1] if spec.selection == 'final' else values
            rows.append({'name':'Linear head', 'tensor':model.head(selected)})
        return model, rows


def cell_trace(x, w_ih, w_hh, b_ih=None, b_hh=None, h0=None, c0=None):
    """Unbatched [T,I] explicit recurrence equivalent to nn.LSTMCell."""
    if x.ndim != 2 or len(x) < 1 or w_hh.ndim != 2 or w_ih.ndim != 2:
        raise ValueError('Expected [T,I] input and rank-two weights')
    hsize = w_hh.shape[1]
    if hsize < 1 or w_hh.shape != (4*hsize,hsize) or w_ih.shape != (4*hsize,x.shape[1]):
        raise ValueError('Packed weights must have shapes [4H,I] and [4H,H]')
    h = x.new_zeros(hsize) if h0 is None else h0
    c = x.new_zeros(hsize) if c0 is None else c0
    bi = x.new_zeros(4*hsize) if b_ih is None else b_ih
    bh = x.new_zeros(4*hsize) if b_hh is None else b_hh
    if h.shape != (hsize,) or c.shape != (hsize,) or bi.shape != (4*hsize,) or bh.shape != (4*hsize,):
        raise ValueError('Initial states must be [H]; biases must be [4H]')
    if not all(torch.isfinite(v).all() for v in (x,w_ih,w_hh,h,c,bi,bh)):
        raise ValueError('All values must be finite')
    steps = []
    for xt in x:
        previous_h, previous_c = h, c
        incoming, recurrent = w_ih @ xt, w_hh @ h
        z = incoming + bi + recurrent + bh
        zi,zf,zg,zo = z.chunk(4)
        i,f,g,o = zi.sigmoid(),zf.sigmoid(),zg.tanh(),zo.sigmoid()
        retained, written = f*c,i*g
        c = retained+written
        h = o*c.tanh()
        steps.append(dict(input_vector=xt, previous_h=previous_h, previous_c=previous_c,
            input_contribution=incoming,recurrent_contribution=recurrent,preactivation=z,
            input=i,forget=f,candidate=g,output=o,retained=retained,written=written,cell=c,hidden=h))
    return {name:torch.stack([s[name] for s in steps]) for name in (*GATES,'cell','hidden')}, steps


def demo_cell(input_size=2, hidden_size=3, sequence_length=6, seed=42, scale=.5):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(sequence_length,input_size,generator=gen),
            torch.randn(4*hidden_size,input_size,generator=gen)*scale,
            torch.randn(4*hidden_size,hidden_size,generator=gen)*scale,
            torch.zeros(4*hidden_size),torch.zeros(4*hidden_size))


def controlled_memory(length=12, forget=.95, input_gate=.5, output_gate=.8, initial=1., reverse=False):
    """Fixed gate interventions, explicitly distinct from learned gates."""
    if type(length) is not int or not 2 <= length <= 100 or not all(math.isfinite(v) for v in (forget,input_gate,output_gate,initial)):
        raise ValueError('Use length 2..100 and finite controls')
    if any(not 0 <= v <= 1 for v in (forget,input_gate,output_gate)):
        raise ValueError('Gate interventions must be in [0,1]')
    x = torch.zeros(length,1,dtype=torch.float64); x[0] = 1.; x[length//2] = -.5
    if reverse: x = x.flip(0)
    c = torch.tensor([initial],dtype=torch.float64,requires_grad=True)
    states, hidden = [], []
    for xt in x:
        c = forget*c + input_gate*xt.tanh()
        c.retain_grad(); states.append(c); hidden.append(output_gate*c.tanh())
    loss = states[-1].sum(); loss.backward()
    return dict(input_vector=x, forget=torch.full_like(x,forget),input=torch.full_like(x,input_gate),
                candidate=x.tanh(),output=torch.full_like(x,output_gate),cell=torch.stack(states).detach(),
                hidden=torch.stack(hidden).detach(),cell_gradients=[s.grad.item() for s in states],
                direct_path=[forget**k for k in range(length-1,-1,-1)])


def gradient_experiment(length=12, forget_bias=2., scale=.3, clip=1., learning_rate=.01, optimizer='SGD', seed=42):
    if type(length) is not int or not 2 <= length <= 100 or optimizer not in ('SGD','Adam'):
        raise ValueError('Invalid length or optimizer')
    if not all(math.isfinite(v) for v in (forget_bias,scale,clip,learning_rate)) or scale < 0 or clip <= 0 or learning_rate <= 0:
        raise ValueError('Finite controls and positive clipping/rate required')
    x,wi,wh,bi,bh = demo_cell(1,2,length,seed,scale)
    x = x.double().requires_grad_()
    bi[2:4] = forget_bias
    params = [nn.Parameter(v.double()) for v in (wi,wh,bi,bh)]
    opt = (torch.optim.SGD if optimizer == 'SGD' else torch.optim.Adam)(params,lr=learning_rate)
    opt.zero_grad()
    values,steps = cell_trace(x,*params)
    for step in steps:
        step['cell'].retain_grad(); step['hidden'].retain_grad()
    prediction = steps[-1]['hidden'].sum()
    loss = (prediction-1.).square()
    loss.backward()
    gradients = {state:[0. if s[state].grad is None else float(s[state].grad.norm()) for s in steps]
                 for state in ('cell','hidden')}
    before = [p.detach().clone() for p in params]
    raw = []
    for p in params:
        assert p.grad is not None
        raw.append(p.grad.detach().clone())
    gate_norms = {name:{gate:float(g.chunk(4)[j].norm()) for j,gate in enumerate(GATES)}
                  for name,g in zip(('weight_ih','weight_hh','bias_ih','bias_hh'),raw)}
    norm_before = float(nn.utils.clip_grad_norm_(params,clip))
    clipped = []
    for p in params:
        assert p.grad is not None
        clipped.append(p.grad.detach().clone())
    norm_after = float(torch.stack([g.square().sum() for g in clipped]).sum().sqrt())
    opt.step()
    after = [p.detach().clone() for p in params]
    updated,_ = cell_trace(x.detach(),*params)
    loss_after = float((updated['hidden'][-1].sum()-1.).square().detach())
    return dict(values={k:v.detach() for k,v in values.items()},steps=steps,prediction=float(prediction.detach()),
                loss=float(loss.detach()),loss_after=loss_after,gradients=gradients,input_gradients=x.grad.detach(),
                gate_norms=gate_norms,before=before,raw_gradients=raw,clipped_gradients=clipped,after=after,
                norm_before=norm_before,norm_after=norm_after)


def variable_length_demo(lengths=(5,2,4), hidden_size=3, seed=42):
    lengths = torch.as_tensor(lengths)
    if lengths.ndim != 1 or not len(lengths) or lengths.dtype not in (torch.int32,torch.int64) or (lengths<1).any() or (lengths>30).any():
        raise ValueError('Integer lengths 1..30 required')
    model = LSTMAdapter().build(LSTMModelSpec(hidden_size=hidden_size,seed=seed))
    x = torch.randn(len(lengths),int(lengths.max()),1,generator=torch.Generator().manual_seed(seed))
    mask = torch.arange(x.shape[1])[None,:] < lengths[:,None]; x[~mask] = 0
    with torch.no_grad():
        padded,_ = model.encode(x)
        output,(hn,cn) = model.encode(x,lengths)
    return dict(model=model,input=x,lengths=lengths,mask=mask,output=output,hidden=hn,cell=cn,
                final_valid=output[torch.arange(len(lengths)),lengths-1],wrong_padded_last=padded[:,-1])
