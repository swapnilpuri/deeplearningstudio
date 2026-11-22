import math
import pytest
import torch
from learning.attention_adapter import visibility, self_attention
from learning.transformer_spec import TransformerSpec
from learning.transformer_adapter import (MultiHeadAttention, TransformerBlock, FeedForward,
    residual, layer_norm_trace, parameter_counts, complexity)


@pytest.mark.parametrize('kwargs', [dict(d_model=7), dict(num_heads=0), dict(num_heads=True),
    dict(d_ff=129), dict(epsilon=0), dict(epsilon=float('nan')), dict(activation='bad'), dict(norm='bad')])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): TransformerSpec(**kwargs)


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('heads', [1, 2, 4])
def test_pytorch_reference(causal, heads):
    spec = TransformerSpec(num_heads=heads)
    model = MultiHeadAttention(spec).double()
    reference = torch.nn.MultiheadAttention(8, heads, bias=False, batch_first=True).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([model.Wq.T, model.Wk.T, model.Wv.T]))
        reference.out_proj.weight.copy_(model.Wo.T)
    x = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    mask = visibility(4, causal, torch.tensor([False, False, False, True]))
    actual = model(x, mask)
    expected, probabilities = reference(x, x, x, attn_mask=~mask, average_attn_weights=False)
    torch.testing.assert_close(actual['output'], expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual['probabilities'], probabilities, atol=1e-10, rtol=1e-10)
    assert actual['scores'].shape == (2, heads, 4, 4)
    assert actual['Q'].shape == (2, heads, 4, 8 // heads)
    torch.testing.assert_close(actual['probabilities'].sum(-1), torch.ones(2, heads, 4, dtype=x.dtype))
    assert torch.count_nonzero(actual['probabilities'].masked_select(~mask)) == 0
    torch.testing.assert_close(actual['concat'], torch.cat(list(actual['heads'].unbind(1)), -1))
    torch.testing.assert_close(actual['output'], actual['concat'] @ model.Wo)
    torch.testing.assert_close(actual['scaled_scores'], actual['Q'] @ actual['K'].transpose(-1,-2) / math.sqrt(spec.d_head))
    if heads == 1:
        old = self_attention(x[0], dict(Wq=model.Wq, Wk=model.Wk, Wv=model.Wv), mask)
        torch.testing.assert_close(old['output'], actual['concat'][0])


def test_determinism_and_bad_inputs():
    torch.manual_seed(17)
    state = torch.random.get_rng_state().clone()
    a, b = TransformerBlock(TransformerSpec()), TransformerBlock(TransformerSpec())
    assert torch.equal(state, torch.random.get_rng_state())
    for x,y in zip(a.parameters(), b.parameters()): assert torch.equal(x,y)
    x = torch.ones(1,4,8)
    for bad in (torch.ones(4,8), torch.ones(1,4,7), torch.full_like(x,float('nan')), torch.ones(1,25,8)):
        with pytest.raises(ValueError): a.mha(bad)
    for mask in (torch.zeros(4,4,dtype=torch.bool), torch.ones(4,4), torch.ones(3,3,dtype=torch.bool)):
        with pytest.raises(ValueError): a.mha(x,mask)
    batched = visibility(4).expand(1,-1,-1)
    torch.testing.assert_close(a.mha(x,batched)['output'], a.mha(x)['output'])
    with pytest.raises(ValueError): residual(x, x[:,:,:1])


@pytest.mark.parametrize('epsilon', [1e-5, .1])
def test_layernorm(epsilon):
    x = torch.tensor([[[1.,2.,3.,4.],[3.,3.,3.,3.]]])
    layer = torch.nn.LayerNorm(4,eps=epsilon)
    with torch.no_grad(): layer.weight.fill_(2); layer.bias.fill_(.5)
    trace = layer_norm_trace(x,layer)
    torch.testing.assert_close(trace['variance'], torch.tensor([[[1.25],[0.]]]))
    torch.testing.assert_close(trace['output'], trace['normalized'] * 2 + .5)
    assert torch.isfinite(trace['output']).all()


@pytest.mark.parametrize('norm', ['Pre-LayerNorm', 'Post-LayerNorm'])
@pytest.mark.parametrize('activation', ['ReLU', 'GELU'])
def test_block_equations_and_gradients(norm, activation):
    block = TransformerBlock(TransformerSpec(norm=norm, activation=activation))
    x = torch.randn(2,4,8,requires_grad=True)
    mask = visibility(4,True)
    trace = block(x,mask)
    if norm == 'Pre-LayerNorm':
        u = x + block.mha(block.ln1(x),mask)['output']
        expected = u + block.ffn(block.ln2(u))['output']
    else:
        u = block.ln1(x + block.mha(x,mask)['output'])
        expected = block.ln2(u + block.ffn(u)['output'])
    torch.testing.assert_close(trace['output'], expected)
    assert trace['ffn']['linear1'].shape == (2,4,16)
    torch.testing.assert_close(block.ffn(x)['output'][:,0],block.ffn(x[:,0])['output'])
    (trace['output'] * torch.randn_like(x)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in block.parameters(): assert p.grad is not None and torch.isfinite(p.grad).all()
    counts = parameter_counts(block)
    assert counts == dict(Wq=64,Wk=64,Wv=64,Wo=64,FFN=280,LayerNorm=32,total=568)
    assert complexity(block.spec,4) == dict(score_cells=32,QK=128,AV=128,projections=1024,FFN=1024)
    changed = x.detach().clone(); changed[:,2:] += 1
    torch.testing.assert_close(block(changed,mask)['output'][:,:2], trace['output'].detach()[:,:2])
