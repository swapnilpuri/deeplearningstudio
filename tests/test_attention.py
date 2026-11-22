import math
import pytest
import torch
from learning.attention_spec import AttentionSpec
from learning.attention_adapter import projections, visibility, attention, self_attention
from learning.attention_training import AttentionTrainingConfig, train_attention


def test_hand_calculation_and_shapes():
    x = torch.tensor([[1.,0.],[0.,1.]])
    w = projections(AttentionSpec(2,2,1,initialization='identity'))
    trace = self_attention(x,w)
    assert torch.equal(trace['Q'],x) and torch.equal(trace['K'],x)
    assert torch.equal(trace['V'],x[:,:1])
    assert torch.equal(trace['scores'],torch.eye(2))
    p = math.exp(1/math.sqrt(2))/(math.exp(1/math.sqrt(2))+1)
    torch.testing.assert_close(trace['probabilities'],torch.tensor([[p,1-p],[1-p,p]]))
    torch.testing.assert_close(trace['output'],torch.tensor([[p],[1-p]]))
    assert trace['output'].shape == (2,1)
    raw = self_attention(x,w,scaled=False)
    torch.testing.assert_close(raw['probabilities'],torch.softmax(torch.eye(2),-1))


def test_causal_padding_and_fully_masked():
    mask = visibility(3, True, torch.tensor([False,False,True]))
    assert torch.equal(mask,torch.tensor([[1,0,0],[1,1,0],[1,1,0]],dtype=torch.bool))
    t = attention(torch.ones(3,2),torch.ones(3,2),torch.ones(3,4),mask)
    assert (t['probabilities'][~mask] == 0).all()
    torch.testing.assert_close(t['probabilities'].sum(-1),torch.ones(3))
    with pytest.raises(ValueError,match='Fully masked'):
        attention(torch.ones(3,2),torch.ones(3,2),torch.ones(3,4),torch.zeros(3,3,dtype=torch.bool))
    for bad in (torch.ones(3,3),torch.ones(2,2,dtype=torch.bool)):
        with pytest.raises(ValueError,match='mask'):
            attention(torch.ones(3,2),torch.ones(3,2),torch.ones(3,4),bad)


def test_determinism_gradients_stability_and_reference():
    w = projections(AttentionSpec())
    assert all(torch.equal(v,projections(AttentionSpec())[k]) for k,v in w.items())
    x = torch.eye(4,requires_grad=True)
    w = {k:v.requires_grad_() for k,v in w.items()}
    t = self_attention(x,w,visibility(4,True))
    reference = torch.nn.functional.scaled_dot_product_attention(t['Q'],t['K'],t['V'],attn_mask=t['visibility'])
    torch.testing.assert_close(t['output'],reference)
    t['output'].square().sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in [x,*w.values()])
    large = attention(torch.full((2,3),1e4),torch.full((2,3),1e4),torch.ones(2,1))
    assert torch.isfinite(large['output']).all()
    torch.testing.assert_close(large['probabilities'],torch.full((2,2),.5))


@pytest.mark.parametrize('kwargs',[{'key_dim':0},{'embedding_dim':33},{'seed':-1},{'initialization':'bad'}])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError): AttentionSpec(**kwargs)


@pytest.mark.parametrize('optimizer',['SGD','Adam'])
def test_training_updates_and_reset(optimizer):
    run = train_attention(AttentionTrainingConfig(optimizer=optimizer))
    assert run['losses'][-1] < run['losses'][0] * (.3 if optimizer=='Adam' else 1)
    for name in run['initial']:
        assert torch.isfinite(run['gradients'][name]).all()
        assert not torch.equal(run['initial'][name],run['trained'][name])
    reset = projections(AttentionSpec(4,3,2,run['config'].seed))
    assert all(torch.equal(reset[k],v) for k,v in run['initial'].items())
    again = train_attention(AttentionTrainingConfig(steps=2,optimizer=optimizer))
    assert again['losses'] == train_attention(AttentionTrainingConfig(steps=2,optimizer=optimizer))['losses']


@pytest.mark.parametrize('kwargs',[{'steps':0},{'steps':501},{'learning_rate':float('nan')},{'optimizer':'bad'}])
def test_invalid_training(kwargs):
    with pytest.raises(ValueError): AttentionTrainingConfig(**kwargs)


def test_invalid_inputs():
    w = projections(AttentionSpec())
    for x in (torch.empty(0,4),torch.ones(25,4),torch.full((2,4),float('nan')),torch.ones(2,3)):
        with pytest.raises(ValueError): self_attention(x,w)
    with pytest.raises(ValueError): visibility(0)
    with pytest.raises(ValueError): visibility(2,padding=torch.ones(2))
    with pytest.raises(ValueError): attention(torch.ones(2,3),torch.ones(2,4),torch.ones(2,1))
