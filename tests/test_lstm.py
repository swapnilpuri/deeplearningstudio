import inspect
import json
import math
import pytest
import torch
from learning.lstm_spec import LSTMModelSpec
from learning.lstm_adapter import LSTMAdapter, cell_trace, demo_cell, controlled_memory, gradient_experiment, variable_length_demo
from learning.rnn_adapter import masked_mse
from learning.lstm_training import LSTMTrainingConfig, lstm_splits, train_lstm, compare_models, dependency_sweep, evaluate_sequence_test


@pytest.mark.parametrize('i,h,t',[(1,1,2),(2,3,7),(3,5,4)])
@pytest.mark.parametrize('bias',[True,False])
def test_cell_reference_and_gradient_parity(i,h,t,bias):
    x,wi,wh,bi,bh = [v.double() for v in demo_cell(i,h,t)]
    bi += .13; bh -= .04
    x.requires_grad_(); wi.requires_grad_(); wh.requires_grad_()
    h0 = torch.full((h,),.2,dtype=torch.float64); c0 = h0+.3
    values,steps = cell_trace(x,wi,wh,bi if bias else None,bh if bias else None,h0,c0)
    ref = torch.nn.LSTMCell(i,h,bias=bias).double()
    with torch.no_grad():
        ref.weight_ih.copy_(wi);ref.weight_hh.copy_(wh)
        if bias:ref.bias_ih.copy_(bi);ref.bias_hh.copy_(bh)
    xr = x.detach().clone().requires_grad_(); hr,cr = h0,c0
    for index,xt in enumerate(xr):
        hr,cr = ref(xt,(hr,cr))
        torch.testing.assert_close(values['hidden'][index],hr)
        torch.testing.assert_close(values['cell'][index],cr)
    values['hidden'][-1].sum().backward();hr.sum().backward()
    torch.testing.assert_close(x.grad,xr.grad)
    torch.testing.assert_close(wi.grad,ref.weight_ih.grad)
    torch.testing.assert_close(wh.grad,ref.weight_hh.grad)


@pytest.mark.parametrize('layers',[1,3])
@pytest.mark.parametrize('batch_first',[True,False])
@pytest.mark.parametrize('selection',['final','sequence'])
@pytest.mark.parametrize('bias',[True,False])
def test_shapes_count_serialization_trace(layers,batch_first,selection,bias):
    spec = LSTMModelSpec(input_size=2,hidden_size=4,num_layers=layers,output_size=3,
                         batch_first=batch_first,selection=selection,bias=bias,dropout=.2 if layers>1 else 0.)
    assert LSTMModelSpec.from_dict(json.loads(json.dumps(spec.as_dict()))) == spec
    x = torch.randn((2,5,2) if batch_first else (5,2,2))
    model,trace = LSTMAdapter().trace(spec,x)
    assert sum(p.numel() for p in model.parameters()) == spec.parameter_count
    assert tuple(model(x).shape) == spec.infer_shapes(2,5)[-1]['shape']
    torch.testing.assert_close(trace[-1]['tensor'],model(x))
    reference = torch.nn.LSTM(2,4,layers,bias=bias,batch_first=batch_first).eval()
    reference.load_state_dict(model.rnn.state_dict())
    output,(hn,cn) = reference(x)
    assert hn.shape == cn.shape == (layers,2,4)
    torch.testing.assert_close(model(x),model.head(hn[-1] if selection=='final' else output))


def test_invalid_and_framework_neutral():
    import learning.lstm_spec as module
    assert 'import torch' not in inspect.getsource(module)
    assert LSTMModelSpec(input_size=2,hidden_size=3).infer_shapes()[1]['parameters'] == 84
    for kwargs in ({'hidden_size':0},{'dropout':1.},{'dropout':.2},{'selection':'bad'},{'schema_version':3}):
        with pytest.raises(ValueError):LSTMModelSpec(**kwargs)
    with pytest.raises(ValueError):LSTMModelSpec.from_dict({'schema_version':3,'architecture':'rnn'})
    args = list(demo_cell());args[1] = torch.zeros(3,2)
    with pytest.raises(ValueError):cell_trace(*args)


def test_controlled_direct_path_and_optimizer_updates():
    for f in (0.,.3,.99,1.):
        result = controlled_memory(forget=f)
        torch.testing.assert_close(torch.tensor(result['cell_gradients']),torch.tensor(result['direct_path']))
    assert controlled_memory(forget=.99)['cell_gradients'][0] > controlled_memory(forget=.3)['cell_gradients'][0]
    a = controlled_memory(output_gate=0.);b = controlled_memory(output_gate=1.)
    torch.testing.assert_close(a['cell'],b['cell']);assert a['hidden'].count_nonzero()==0
    for optimizer in ('SGD','Adam'):
        r = gradient_experiment(clip=.001,optimizer=optimizer)
        assert r['norm_before'] > .001 and r['norm_after'] <= .001001
        assert all(math.isfinite(v) for values in r['gradients'].values() for v in values)
        assert any(v>0 for v in r['gradients']['cell']) and any(v>0 for v in r['gradients']['hidden'])
        assert math.isfinite(r['loss_after'])
        for old,g,new in zip(r['before'],r['clipped_gradients'],r['after']):
            if optimizer=='SGD':torch.testing.assert_close(new,old-.01*g)


def test_packed_states_independent_execution_and_padding_mask():
    r = variable_length_demo()
    torch.testing.assert_close(r['final_valid'],r['hidden'][-1])
    for j,length in enumerate(r['lengths']):
        _,(hn,cn) = r['model'].encode(r['input'][j:j+1,:length])
        torch.testing.assert_close(hn[:,0],r['hidden'][:,j])
        torch.testing.assert_close(cn[:,0],r['cell'][:,j])
    poisoned = r['input'].clone();poisoned[~r['mask']] = 1e6
    out,(hn,cn) = r['model'].encode(poisoned,r['lengths'])
    torch.testing.assert_close(hn,r['hidden']);torch.testing.assert_close(cn,r['cell'])
    p = torch.ones(3,5,1,requires_grad=True);target = torch.zeros_like(p);target[~r['mask']] = float('nan')
    loss = masked_mse(p,target,r['lengths']);loss.backward()
    assert loss.item()==1 and p.grad[~r['mask']].count_nonzero()==0


@pytest.mark.parametrize('task',['memory','forecasting'])
@pytest.mark.parametrize('optimizer',['SGD','Adam'])
def test_training_reproducibility_split_isolation_and_heldout(task,optimizer):
    cfg = LSTMTrainingConfig(task=task,samples=60,epochs=2,optimizer=optimizer,num_layers=2,dropout=.2,horizon=2,noise=0.)
    data = lstm_splits(cfg);again = lstm_splits(cfg)
    for part in ('train','validation','test'):
        torch.testing.assert_close(data[part][0],again[part][0])
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(data['source_series'][left]) & set(data['source_series'][right])
    if task=='memory':
        x,y = data['train'];torch.testing.assert_close(x[:,data['signal_position'],0],2*y.float()-1)
    run = train_lstm(cfg,data)
    poison = dict(data);poison['test'] = (torch.full_like(data['test'][0],float('nan')),data['test'][1])
    poison['validation'] = (data['validation'][0]*10,data['validation'][1].flip(0))
    other = train_lstm(cfg,poison)
    for a,b in zip(run['model'].parameters(),other['model'].parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert any(v>0 for v in run['parameter_changes'].values())
    assert all(row['clipped_norm'] <= cfg.clip+1e-5 for row in run['history'])
    assert math.isfinite(evaluate_sequence_test(run)['loss'])


def test_comparison_shared_data_and_measured_sweep():
    cfg = LSTMTrainingConfig(samples=60,epochs=1,hidden_size=3)
    runs = compare_models(cfg)
    assert runs['RNN']['data'] is runs['LSTM']['data']
    assert runs['LSTM']['spec'].parameter_count > runs['RNN']['spec'].parameter_count
    rows = dependency_sweep(cfg,[1,7]);assert len(rows)==4
    assert all(r['seconds']>0 and math.isfinite(r['validation_metric']) for r in rows)
