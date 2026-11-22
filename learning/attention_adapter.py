"""Bias-free row-vector attention. True mask cells are visible; autograd stays intact."""
from collections.abc import Mapping
import math
import torch
from .attention_spec import AttentionSpec


def projections(spec: AttentionSpec) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(spec.seed)
    result = {}
    for name, width in (('Wq', spec.key_dim), ('Wk', spec.key_dim), ('Wv', spec.value_dim)):
        shape = (spec.embedding_dim, width)
        if spec.initialization == 'normal':
            value = torch.randn(shape, generator=generator) / math.sqrt(spec.embedding_dim)
        elif spec.initialization == 'identity':
            value = torch.eye(*shape)
        else:
            value = torch.zeros(shape)
        result[name] = value
    return result


def visibility(length: int, causal: bool = False, padding: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 24:
        raise ValueError('Use 1 to 24 tokens')
    mask = torch.ones(length, length, dtype=torch.bool)
    if causal:
        mask = mask.tril()
    if padding is not None:
        if padding.dtype != torch.bool or padding.shape != (length,):
            raise ValueError('Padding mask must be bool [T], True marks padding keys')
        mask = mask & ~padding[None, :]
    return mask


def _matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or not value.is_floating_point() or not value.numel() or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite floating matrix')


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None, scaled: bool = True) -> dict[str, torch.Tensor]:
    for name, value in (('Q', q), ('K', k), ('V', v)):
        _matrix(value, name)
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError('Q/K widths and K/V sequence lengths must agree')
    if any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
        raise ValueError('Q/K/V must share dtype and device')
    scores = q @ k.T
    scaled_scores = scores / math.sqrt(q.shape[1])
    logits = scaled_scores if scaled else scores
    if not torch.isfinite(logits).all():
        raise ValueError('Score overflow: reduce input magnitudes')
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape or mask.dtype != torch.bool or mask.device != q.device:
        raise ValueError('Visibility mask must be bool [queries,keys] on the same device')
    if not mask.any(dim=-1).all():
        raise ValueError('Fully masked query row: allow at least one key per query')
    masked = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1)
    output = probabilities @ v
    if not torch.isfinite(output).all():
        raise ValueError('Output overflow: reduce input magnitudes')
    return dict(Q=q, K=k, V=v, scores=scores, scaled_scores=scaled_scores,
                masked_logits=masked, probabilities=probabilities, output=output, visibility=mask)


def self_attention(x: torch.Tensor, weights: Mapping[str, torch.Tensor],
                   mask: torch.Tensor | None = None, scaled: bool = True) -> dict[str, torch.Tensor]:
    _matrix(x, 'X')
    if not 1 <= x.shape[0] <= 24 or not 1 <= x.shape[1] <= 32:
        raise ValueError('Use 1â€“24 tokens and 1â€“32 embedding coordinates')
    for name in ('Wq', 'Wk', 'Wv'):
        _matrix(weights[name], name)
        if weights[name].shape[0] != x.shape[1] or weights[name].shape[1] > 32:
            raise ValueError('Projection shape must be [embedding width, output width <=32]')
        if weights[name].dtype != x.dtype or weights[name].device != x.device:
            raise ValueError('Inputs and projections must share dtype and device')
    return dict(X=x, **attention(x @ weights['Wq'], x @ weights['Wk'], x @ weights['Wv'], mask, scaled))
