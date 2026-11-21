import torch
from models.feed_forward import build_model, generate_code
from visualization.network_graph import network_figure

def test_default():
    m=build_model([4,6,4,2]); assert sum(p.numel() for p in m.parameters())==68
    assert m(torch.zeros(1,4)).shape==(1,2)

def test_no_bias_no_hidden():
    m=build_model([3,2],bias=False); assert len(m)==1
    assert sum(p.numel() for p in m.parameters())==6

def test_seed():
    a=build_model([4,2]); b=build_model([4,2]); assert torch.equal(a[0].weight,b[0].weight)

def test_graph_and_code():
    assert len(network_figure([4,6,4,2]).data)==5
    compile(generate_code([4,6,4,2],'GELU',True,42),'generated.py','exec')
