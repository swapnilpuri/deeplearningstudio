"""Differentiable row-vector components; M9 remains the attention source of truth."""
import math
import torch
from torch import nn
from .attention_adapter import attention
from .transformer_spec import TransformerSpec


def residual(x: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
    if x.shape != sublayer.shape:
        raise ValueError('Residual tensors must have identical shapes')
    return x + sublayer


def layer_norm_trace(x: torch.Tensor, layer: nn.LayerNorm) -> dict[str, torch.Tensor]:
    mean = x.mean(-1, keepdim=True)
    variance = x.var(-1, correction=0, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + layer.eps)
    return dict(input=x, mean=mean, variance=variance, normalized=normalized,
                gamma=layer.weight, beta=layer.bias, output=layer(x))


class MultiHeadAttention(nn.Module):
    """Bias-free Q/K/V/O, standard heads. Trace axes are [B,H,T,d_head]."""
    Wq: nn.Parameter
    Wk: nn.Parameter
    Wv: nn.Parameter
    Wo: nn.Parameter

    def __init__(self, spec: TransformerSpec):
        super().__init__()
        self.spec = spec
        generator = torch.Generator().manual_seed(spec.seed)
        for name in ('Wq', 'Wk', 'Wv', 'Wo'):
            self.register_parameter(name, nn.Parameter(torch.randn(
                spec.d_model, spec.d_model, generator=generator) / math.sqrt(spec.d_model)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        d, h, dh = self.spec.d_model, self.spec.num_heads, self.spec.d_head
        if (x.ndim != 3 or not 1 <= x.shape[0] <= 64 or not 1 <= x.shape[1] <= 24
                or x.shape[-1] != d or not x.is_floating_point() or not torch.isfinite(x).all()):
            raise ValueError('X must be finite floating [batch 1–64, tokens 1–24, d_model]')
        if x.device != self.Wq.device or x.dtype != self.Wq.dtype:
            raise ValueError('X and model must share dtype and device')
        b, t, _ = x.shape
        if mask is not None and (mask.dtype != torch.bool or mask.device != x.device
                                 or mask.shape not in ((t, t), (b, t, t))):
            raise ValueError('Visibility must be bool [T,T] or [B,T,T] on the input device')
        q, k, v = [(x @ w).reshape(b, t, h, dh).transpose(1, 2)
                   for w in (self.Wq, self.Wk, self.Wv)]
        batches = []
        for batch in range(b):
            visible = None if mask is None else (mask if mask.ndim == 2 else mask[batch])
            heads = [attention(q[batch, head], k[batch, head], v[batch, head], visible)
                     for head in range(h)]
            batches.append({name: torch.stack([head[name] for head in heads]) for name in heads[0]})
        trace = {name: torch.stack([batch[name] for batch in batches]) for name in batches[0]}
        concat = trace['output'].transpose(1, 2).reshape(b, t, d)
        output = concat @ self.Wo
        if not torch.isfinite(output).all():
            raise ValueError('Output overflow: reduce input magnitudes')
        return dict(X=x, **{name: value for name, value in trace.items() if name != 'output'},
                    heads=trace['output'], concat=concat, output=output)


class FeedForward(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed + 1)
            self.linear1 = nn.Linear(spec.d_model, spec.d_ff)
            self.linear2 = nn.Linear(spec.d_ff, spec.d_model)
        self.activation = nn.ReLU() if spec.activation == 'ReLU' else nn.GELU()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.linear1(x)
        activated = self.activation(hidden)
        return dict(input=x, linear1=hidden, activation=activated, output=self.linear2(activated))


class TransformerBlock(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        self.spec = spec
        self.mha = MultiHeadAttention(spec)
        self.ln1 = nn.LayerNorm(spec.d_model, eps=spec.epsilon)
        self.ln2 = nn.LayerNorm(spec.d_model, eps=spec.epsilon)
        self.ffn = FeedForward(spec)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if self.spec.norm == 'Pre-LayerNorm':
            n1 = layer_norm_trace(x, self.ln1)
            mha = self.mha(n1['output'], mask)
            u = residual(x, mha['output'])
            n2 = layer_norm_trace(u, self.ln2)
            ffn = self.ffn(n2['output'])
            y = residual(u, ffn['output'])
            stages = dict(X=x, LN1=n1['output'], MHA=mha['output'], U=u,
                          LN2=n2['output'], FFN=ffn['output'], Y=y)
        else:
            mha = self.mha(x, mask)
            r1 = residual(x, mha['output'])
            n1 = layer_norm_trace(r1, self.ln1)
            u = n1['output']
            ffn = self.ffn(u)
            r2 = residual(u, ffn['output'])
            n2 = layer_norm_trace(r2, self.ln2)
            y = n2['output']
            stages = dict(X=x, MHA=mha['output'], residual1=r1, U=u,
                          FFN=ffn['output'], residual2=r2, Y=y)
        if not torch.isfinite(y).all():
            raise ValueError('Block overflow: reduce inputs or learning rate')
        return dict(output=y, mha=mha, ln1=n1, ln2=n2, ffn=ffn, stages=stages)


def parameter_counts(block: TransformerBlock) -> dict[str, int]:
    def count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    result = {name: getattr(block.mha, name).numel() for name in ('Wq', 'Wk', 'Wv', 'Wo')}
    result.update(FFN=count(block.ffn), LayerNorm=count(block.ln1) + count(block.ln2))
    result['total'] = count(block)
    return result


def complexity(spec: TransformerSpec, length: int) -> dict[str, int]:
    """Per-sequence multiply-accumulates; excludes softmax/norm/activation/bias."""
    if not 1 <= length <= 24:
        raise ValueError('Use 1–24 tokens')
    d = spec.d_model
    return dict(score_cells=spec.num_heads * length ** 2, QK=length ** 2 * d,
                AV=length ** 2 * d, projections=4 * length * d ** 2,
                FFN=2 * length * d * spec.d_ff)
