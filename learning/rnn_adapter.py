"""Real recurrent execution and autograd, independent of Streamlit."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .rnn_spec import RNNModelSpec


class RecurrentModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.rnn = nn.RNN(spec.input_size, spec.hidden_size, spec.num_layers,
                          nonlinearity=spec.activation, bias=spec.bias, batch_first=spec.batch_first)
        self.head = nn.Linear(spec.hidden_size, spec.output_size, bias=spec.bias)

    def encode(self, x, lengths=None):
        if x.ndim != 3 or x.shape[-1] != self.spec.input_size or not torch.isfinite(x).all():
            raise ValueError('Expected finite rank-three sequence input with matching features')
        if lengths is None:
            return self.rnn(x)
        baxis, taxis = (0,1) if self.spec.batch_first else (1,0)
        lengths = torch.as_tensor(lengths, device='cpu')
        if lengths.dtype not in (torch.int32,torch.int64) or lengths.shape != (x.shape[baxis],) or (lengths < 1).any() or (lengths > x.shape[taxis]).any():
            raise ValueError('One valid integer length per sequence required')
        packed = pack_padded_sequence(x, lengths, batch_first=self.spec.batch_first, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        padded, _ = pad_packed_sequence(output, batch_first=self.spec.batch_first, total_length=x.shape[taxis])
        return padded, hidden

    def forward(self, x, lengths=None):
        output, hidden = self.encode(x, lengths)
        return self.head(hidden[-1] if self.spec.selection == 'final' else output)


class RNNAdapter:
    def build(self, spec: RNNModelSpec):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            return RecurrentModel(spec)

    def trace(self, spec, x):
        model = self.build(spec)
        rows = [{'name':'Sequence input', 'tensor':x.detach()}]
        values = x
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for layer in range(spec.num_layers):
                width = spec.input_size if layer == 0 else spec.hidden_size
                cell = nn.RNN(width, spec.hidden_size, nonlinearity=spec.activation,
                              bias=spec.bias, batch_first=spec.batch_first)
                cell.load_state_dict({key.replace(f'_l{layer}', '_l0'):value
                                      for key,value in model.rnn.state_dict().items() if key.endswith(f'_l{layer}')})
                values, hidden = cell(values)
                rows.append({'name':f'RNN layer {layer+1}', 'tensor':values})
            selected = hidden[-1] if spec.selection == 'final' else values
            rows.extend([{'name':'Selected representation', 'tensor':selected},
                         {'name':'Linear head', 'tensor':model.head(selected)}])
        return model, rows


def cell_trace(x, w_ih, w_hh, b_ih=None, b_hh=None, h0=None, activation='tanh'):
    """Trace one unbatched sequence [T,I]; retain the complete autograd graph."""
    if activation not in ('tanh','relu') or x.ndim != 2 or x.shape[0] < 1:
        raise ValueError('Expected [T,I] sequence and tanh/relu')
    if w_hh.ndim != 2 or w_ih.ndim != 2:
        raise ValueError('Weight matrices must be rank two')
    hsize = w_hh.shape[0]
    if w_ih.shape != (hsize,x.shape[1]) or w_hh.shape != (hsize,hsize):
        raise ValueError('Incompatible recurrent weight shapes')
    h = x.new_zeros(hsize) if h0 is None else h0
    bi = x.new_zeros(hsize) if b_ih is None else b_ih
    bh = x.new_zeros(hsize) if b_hh is None else b_hh
    if any(v.shape != (hsize,) for v in (h,bi,bh)) or not all(torch.isfinite(v).all() for v in (x,w_ih,w_hh,h,bi,bh)):
        raise ValueError('Finite values and matching hidden/bias dimensions required')
    steps = []
    for xt in x:
        previous = h
        incoming, recurrent = w_ih @ xt, w_hh @ previous
        z = incoming + bi + recurrent + bh
        h = torch.tanh(z) if activation == 'tanh' else torch.relu(z)
        steps.append({'input':xt,'previous':previous,'w_ih':w_ih,'w_hh':w_hh,
                      'input_contribution':incoming,'recurrent_contribution':recurrent,
                      'b_ih':bi,'b_hh':bh,'preactivation':z,'hidden':h})
    return torch.stack([step['hidden'] for step in steps]), steps


def demo_cell(input_size=2, hidden_size=3, sequence_length=5, seed=42, scale=.5):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(sequence_length,input_size,generator=generator),
            torch.randn(hidden_size,input_size,generator=generator)*scale,
            torch.randn(hidden_size,hidden_size,generator=generator)*scale,
            torch.zeros(hidden_size), torch.zeros(hidden_size))


def gradient_experiment(sequence_length=12, recurrent_gain=.5, activation='tanh', truncate=0, clip=1., learning_rate=.01):
    """Terminal linear loss isolates chain products; positive ReLU avoids dead units.

    detach occurs BEFORE zero-based positions k,2k,...; last loss cannot reach
    earlier inputs/states, but the same shared parameter receives local gradients.
    """
    if not 2 <= sequence_length <= 100 or activation not in ('tanh','relu') or not 0 <= truncate <= sequence_length:
        raise ValueError('Invalid gradient experiment')
    if not all(math.isfinite(v) for v in (recurrent_gain,clip,learning_rate)) or not 0 <= recurrent_gain <= 2 or clip <= 0 or learning_rate <= 0:
        raise ValueError('Finite gain 0..2 and positive clipping/rate required')
    weight = nn.Parameter(torch.tensor([[recurrent_gain]], dtype=torch.float64))
    x = torch.full((sequence_length,1), .01, dtype=torch.float64, requires_grad=True)
    h = torch.full((1,), .01, dtype=torch.float64)
    states, boundaries = [], []
    for t in range(sequence_length):
        if truncate and t and t % truncate == 0:
            h = h.detach(); boundaries.append(t)
        z = weight @ h + x[t]
        h = torch.tanh(z) if activation == 'tanh' else torch.relu(z)
        h.retain_grad(); states.append(h)
    loss = states[-1].sum()
    loss.backward()
    assert weight.grad is not None and x.grad is not None
    state_norms = [0. if s.grad is None else s.grad.norm().item() for s in states]
    contributions = []
    for t,state in enumerate(states):
        previous = .01 if t == 0 else float(states[t-1].detach().item())
        derivative = 1. if activation == 'relu' else 1.-float(state.detach().item())**2
        contributions.append(0. if state.grad is None else float(state.grad.item())*derivative*previous)
    before = weight.detach().clone()
    gradient_before = weight.grad.detach().clone()
    norm_before = float(nn.utils.clip_grad_norm_([weight],clip))
    norm_after = float(weight.grad.norm())
    optimizer = torch.optim.SGD([weight],lr=learning_rate)
    optimizer.step()
    return {'loss':float(loss.detach()), 'state_gradients':state_norms,
            'input_gradients':x.grad.detach(), 'boundaries':boundaries,'weight_contributions':contributions,
            'norm_before':norm_before,'norm_after':norm_after, 'threshold':clip,
            'before':before,'gradient_before':gradient_before,'gradient_after':weight.grad.detach().clone(),
            'after':weight.detach().clone(),'states':torch.stack(states).detach()}


def variable_length_demo(lengths=(5,2,4), hidden_size=3, seed=42):
    lengths = torch.as_tensor(lengths)
    if lengths.ndim != 1 or not len(lengths) or lengths.dtype not in (torch.int32,torch.int64) or (lengths < 1).any() or (lengths > 30).any():
        raise ValueError('Integer lengths 1..30 required')
    spec = RNNModelSpec(hidden_size=hidden_size,seed=seed)
    model = RNNAdapter().build(spec)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths),int(lengths.max()),1,generator=generator)
    mask = torch.arange(x.shape[1])[None,:] < lengths[:,None]
    x[~mask] = 0
    with torch.no_grad():
        padded, _ = model.encode(x)
        packed, hidden = model.encode(x,lengths)
    final_valid = packed[torch.arange(len(lengths)),lengths-1]
    return {'input':x,'lengths':lengths,'mask':mask,'padded_output':padded,'packed_output':packed,
            'final_valid':final_valid,'packed_hidden':hidden[-1],'wrong_padded_last':padded[:,-1]}


def masked_mse(prediction, target, lengths):
    """Supervise only valid [B,T,F] positions; padding has zero gradient."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Matching [B,T,F] tensors required')
    lengths = torch.as_tensor(lengths,device=prediction.device)
    if lengths.shape != (prediction.shape[0],) or lengths.dtype not in (torch.int32,torch.int64) or (lengths < 1).any() or (lengths > prediction.shape[1]).any():
        raise ValueError('Invalid lengths')
    mask = torch.arange(prediction.shape[1],device=prediction.device)[None,:] < lengths[:,None]
    return (prediction[mask]-target[mask]).square().mean()
