"""Inspectable encoder-decoder; M8 positions, M9 attention, M10 projections/FFN.

Tensor convention: [batch, tokens, features]; attention [batch, heads, query, key].
Padding masks use True=PAD; visibility matrices use True=permitted.
"""
from dataclasses import replace
import torch
from torch import nn
from .embedding_adapter import sinusoidal_positions, validate_ids
from .attention_adapter import attention
from .transformer_adapter import MultiHeadAttention, FeedForward, residual
from .seq2seq_spec import Seq2SeqSpec


def attention_visibility(query_length: int, padding: torch.Tensor, causal: bool = False):
    if (isinstance(query_length, bool) or not isinstance(query_length, int)
            or not 1 <= query_length <= 24 or padding.dtype != torch.bool
            or padding.ndim != 2 or not 1 <= padding.shape[0] <= 64
            or not 1 <= padding.shape[1] <= 24):
        raise ValueError('Expected query length 1–24 and bool padding [B 1–64,K 1–24]')
    visible = (~padding[:, None, :]).expand(-1, query_length, -1)
    if causal:
        if padding.shape[1] != query_length:
            raise ValueError('Causal self-attention requires equal query/key lengths')
        visible = visible & torch.ones(query_length, query_length, dtype=torch.bool,
                                       device=padding.device).tril()
    if not visible.any(-1).all():
        raise ValueError('Fully masked rows are invalid; include a non-PAD source and start targets with BOS')
    return visible


class PositionalInput(nn.Module):
    fixed: torch.Tensor
    def __init__(self, spec: Seq2SeqSpec, vocabulary: int):
        super().__init__()
        self.spec = spec
        self.tokens = nn.Embedding(vocabulary, spec.block.d_model, padding_idx=0)
        self.positions = nn.Embedding(spec.max_length, spec.block.d_model) if spec.positional == 'learned' else None
        self.register_buffer('fixed', sinusoidal_positions(spec.max_length, spec.block.d_model))
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, ids):
        validate_ids(ids, self.tokens.num_embeddings)
        if ids.ndim != 2 or not 1 <= ids.shape[0] <= 64 or not 1 <= ids.shape[1] <= self.spec.max_length:
            raise ValueError('IDs must be [batch 1–64, sequence 1–max_length]')
        token = self.tokens(ids)
        position = (self.positions(torch.arange(ids.shape[1], device=ids.device))
                    if self.positions is not None else self.fixed[:ids.shape[1]])
        position = position[None].expand_as(token)
        return dict(token=token, position=position, combined=token + position,
                    output=self.dropout(token + position))


class CrossAttention(nn.Module):
    """M10 packed weights, with decoder Q and encoder K/V passed to M9 attention."""
    Wq: nn.Parameter
    Wk: nn.Parameter
    Wv: nn.Parameter
    Wo: nn.Parameter

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        projections = MultiHeadAttention(spec)
        for name in ('Wq', 'Wk', 'Wv', 'Wo'):
            self.register_parameter(name, getattr(projections, name))

    def forward(self, x, memory, mask=None):
        b, t, d = x.shape
        s = memory.shape[1]
        if memory.shape[0] != b or memory.shape[-1] != d or d != self.spec.d_model:
            raise ValueError('Decoder and memory batch/model dimensions must agree')
        h, dh = self.spec.num_heads, self.spec.d_head
        q = (x @ self.Wq).reshape(b, t, h, dh).transpose(1, 2)
        k = (memory @ self.Wk).reshape(b, s, h, dh).transpose(1, 2)
        v = (memory @ self.Wv).reshape(b, s, h, dh).transpose(1, 2)
        if mask is not None and (mask.shape != (b, t, s) or mask.dtype != torch.bool):
            raise ValueError('Cross visibility must be bool [B,target,source]')
        rows = [[attention(q[i, j], k[i, j], v[i, j], None if mask is None else mask[i])
                 for j in range(h)] for i in range(b)]
        trace = {name: torch.stack([torch.stack([head[name] for head in row]) for row in rows])
                 for name in rows[0][0]}
        heads = trace.pop('output')
        concat = heads.transpose(1, 2).reshape(b, t, d)
        return dict(**trace, heads=heads, concat=concat, output=concat @ self.Wo)


class ResidualLayer(nn.Module):
    def __init__(self, spec: Seq2SeqSpec):
        super().__init__()
        self.pre = spec.block.norm == 'Pre-LayerNorm'
        self.self_attention = MultiHeadAttention(spec.block)
        self.ffn = FeedForward(spec.block)
        self.norms = nn.ModuleList([nn.LayerNorm(spec.block.d_model, eps=spec.block.epsilon) for _ in range(2)])
        self.dropout = nn.Dropout(spec.dropout)

    def _step(self, x, y, index):
        result = residual(x, self.dropout(y))
        return result if self.pre else self.norms[index](result)


class EncoderLayer(ResidualLayer):
    def forward(self, x, mask):
        stages = dict(input=x)
        attention_input = self.norms[0](x) if self.pre else x
        a = self.self_attention(attention_input, mask)
        u = self._step(x, a['output'], 0)
        ffn_input = self.norms[1](u) if self.pre else u
        f = self.ffn(ffn_input)
        y = self._step(u, f['output'], 1)
        stages.update(attention_input=attention_input, self_attention=a['output'], residual_attention=u,
                      ffn_input=ffn_input, ffn_hidden=f['linear1'], ffn_activated=f['activation'], ffn=f['output'], output=y)
        return dict(output=y, self_attention=a, ffn=f, stages=stages)


class DecoderLayer(ResidualLayer):
    def __init__(self, spec: Seq2SeqSpec):
        super().__init__(spec)
        self.cross_attention = CrossAttention(replace(spec.block, seed=spec.block.seed + 17))
        self.norms.append(nn.LayerNorm(spec.block.d_model, eps=spec.block.epsilon))

    def forward(self, x, memory, self_mask, memory_mask):
        attention_input = self.norms[0](x) if self.pre else x
        a = self.self_attention(attention_input, self_mask)
        u = self._step(x, a['output'], 0)
        cross_input = self.norms[1](u) if self.pre else u
        c = self.cross_attention(cross_input, memory, memory_mask)
        v = self._step(u, c['output'], 1)
        ffn_input = self.norms[2](v) if self.pre else v
        f = self.ffn(ffn_input)
        y = self._step(v, f['output'], 2)
        return dict(output=y, self_attention=a, cross_attention=c, ffn=f,
                    stages=dict(input=x, attention_input=attention_input, masked_self_attention=a['output'], residual_self=u,
                                cross_input=cross_input, cross_attention=c['output'], residual_cross=v,
                                ffn_input=ffn_input, ffn_hidden=f['linear1'], ffn_activated=f['activation'], ffn=f['output'], output=y))


class EncoderDecoder(nn.Module):
    def __init__(self, spec: Seq2SeqSpec):
        super().__init__()
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.block.seed)
            self.source = PositionalInput(spec, spec.source_vocabulary)
            self.target = PositionalInput(spec, spec.target_vocabulary)
            self.encoder = nn.ModuleList([EncoderLayer(replace(spec, block=replace(
                spec.block, seed=spec.block.seed + 101 * i))) for i in range(spec.encoder_layers)])
            self.decoder = nn.ModuleList([DecoderLayer(replace(spec, block=replace(
                spec.block, seed=spec.block.seed + 1000 + 101 * i))) for i in range(spec.decoder_layers)])
            self.projection = nn.Linear(spec.block.d_model, spec.target_vocabulary)

    def encode(self, source):
        embedding = self.source(source)
        mask = attention_visibility(source.shape[1], source == 0)
        x, layers = embedding['output'], []
        for layer in self.encoder:
            trace = layer(x, mask)
            layers.append(trace)
            x = trace['output']
        return dict(embedding=embedding, layers=layers, memory=x, visibility=mask)

    def decode(self, target, memory, source_padding):
        embedding = self.target(target)
        if target.shape[0] != memory.shape[0] or source_padding.shape != memory.shape[:2]:
            raise ValueError('Source memory, padding, and target batches must agree')
        self_mask = attention_visibility(target.shape[1], target == 0, causal=True)
        cross_mask = attention_visibility(target.shape[1], source_padding)
        x, layers = embedding['output'], []
        for layer in self.decoder:
            trace = layer(x, memory, self_mask, cross_mask)
            layers.append(trace)
            x = trace['output']
        return dict(embedding=embedding, layers=layers, output=x, logits=self.projection(x),
                    self_visibility=self_mask, cross_visibility=cross_mask)

    def inspect(self, source, target):
        encoder = self.encode(source)
        decoder = self.decode(target, encoder['memory'], source == 0)
        return dict(encoder=encoder, decoder=decoder, logits=decoder['logits'])

    def forward(self, source, target):
        return self.inspect(source, target)['logits']


def parameter_counts(model):
    """Disjoint categories, derived from actual trainable tensors."""
    counts = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split('.')
        group = '.'.join(parts[:3]) if parts[0] in ('encoder', 'decoder') else '.'.join(parts[:2])
        counts[group] = counts.get(group, 0) + parameter.numel()
    counts['total'] = sum(counts.values())
    return counts


def complexity(spec, source_length, target_length):
    if not all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= spec.max_length
               for x in (source_length, target_length)):
        raise ValueError('Lengths must be within the configured sequence limit')
    h, d = spec.block.num_heads, spec.block.d_model
    e, r = spec.encoder_layers, spec.decoder_layers
    return dict(encoder_score_cells=e*h*source_length**2,
                decoder_score_cells=r*h*target_length**2,
                cross_score_cells=r*h*target_length*source_length,
                attention_QK_AV_MACs=2*d*(e*source_length**2+r*target_length**2+r*source_length*target_length))
