import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from learning.cnn_spec import CNNLayerSpec, CNNModelSpec, digits_spec, output_axis
from learning.cnn_adapter import CNNAdapter, convolution_explorer, pooling_explorer, single_sgd_step
from learning.cnn_training import CNNTrainingConfig, digits_splits, train_digits, evaluate_test


def test_spec_roundtrip_shapes_and_params():
    spec=digits_spec()
    assert CNNModelSpec.from_dict(json.loads(json.dumps(spec.as_dict())))==spec
    rows=spec.infer_shapes()
    assert rows[-1]['shape']==(1,10)
    assert rows[1]['weights']==(8,1,3,3)
    assert rows[-1]['weights']==(10,64)
    model=CNNAdapter().build(spec)
    assert sum(p.numel() for p in model.parameters())==sum(r['parameters'] for r in rows)
    assert tuple(model(torch.rand(3,1,8,8)).shape)==(3,10)


def test_output_formula_and_invalid_specs():
    assert output_axis(8,3,1,1)==8
    assert output_axis(8,3,2,0)==3
    assert output_axis(8,3,1,0,2)==4
    with pytest.raises(ValueError):output_axis(3,5)
    with pytest.raises(ValueError):CNNModelSpec(1,8,8,(CNNLayerSpec('conv2d',out_channels=8,kernel_size=(11,11)),CNNLayerSpec('flatten'),CNNLayerSpec('linear',out_features=10)))
    with pytest.raises(ValueError):CNNModelSpec.from_dict({'schema_version':1,'architecture':'feed_forward'})


@pytest.mark.parametrize('cin,out,stride,padding,bias',[ (1,1,1,0,False),(1,3,2,1,True),(3,2,1,1,True)])
def test_conv_matches_torch(cin,out,stride,padding,bias):
    torch.manual_seed(9)
    x=torch.randn(cin,7,7);w=torch.randn(out,cin,3,3);b=torch.randn(out) if bias else None
    ref=F.conv2d(x[None],w,b,stride=stride,padding=padding)[0]
    result=convolution_explorer(x,w,b,stride,padding,'ReLU',0,0,0)
    torch.testing.assert_close(result['output'],ref)
    assert result['linear_value']==pytest.approx(ref[0,0,0].item(),abs=1e-5)
    assert result['sum_products']+result['bias']==pytest.approx(ref[0,0,0].item(),abs=1e-5)
    torch.testing.assert_close(result['activated'],F.relu(ref))


def test_cross_correlation_known_example():
    x=torch.arange(1,17,dtype=torch.float32).reshape(1,4,4)
    w=torch.tensor([[[[1.,0.],[0.,-1.]]]])
    r=convolution_explorer(x,w)
    assert r['linear_value']==-5.0
    assert r['output'].shape==(1,3,3)


@pytest.mark.parametrize('kind',["max","average"])
def test_pool_values(kind):
    x=torch.arange(16,dtype=torch.float32).reshape(4,4)
    r=pooling_explorer(x,kind,2,2)
    ref=F.max_pool2d(x[None,None],2,2) if kind=='max' else F.avg_pool2d(x[None,None],2,2)
    torch.testing.assert_close(r['output'],ref[0,0]);assert r['window'].shape==(2,2)


def test_backward_sgd_and_gradient_check():
    spec=digits_spec(filters1=2,filters2=2)
    x=torch.rand(1,1,8,8);target=torch.tensor([5])
    result=single_sgd_step(spec,x,target,.001)
    assert any(v['gradient_norm']>0 for v in result['parameters'].values())
    for data in result['parameters'].values():
        torch.testing.assert_close(data['after'],data['before']-.001*data['gradient'],rtol=1e-5,atol=1e-6)
    # Independent finite differences of a double-precision Conv2d input.
    inp=torch.randn(1,1,4,4,dtype=torch.double,requires_grad=True)
    kernel=torch.randn(1,1,2,2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:F.conv2d(a,b).sum(),(inp,kernel),eps=1e-6,atol=1e-4)


def test_splits_no_overlap_and_normalized():
    data=digits_splits(42);ids=data['indices']
    assert all(not(set(ids[a])&set(ids[b])) for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert sum(len(v) for v in ids.values())==1797
    for part in ('train','validation','test'):
        x,y=data[part]
        assert x.shape[1:]==(1,8,8) and x.min()>=0 and x.max()<=1
        assert y.dtype==torch.int64


def test_training_smoke_and_heldout_evaluation():
    run=train_digits(CNNTrainingConfig(epochs=1,batch_size=128,learning_rate=.01,filters1=2,filters2=2))
    assert len(run['history'])==1
    assert np.isfinite(run['history'][0]['train_loss'])
    assert run['history'][0]['conv_gradient_norm']>0
    result=evaluate_test(run)
    assert result['confusion_matrix'].shape==(10,10)
    assert 0<=result['accuracy']<=1
