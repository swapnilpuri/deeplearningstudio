import inspect
import json
import math
import numpy as np
import pytest
import torch
from learning.rnn_spec import RNNModelSpec
from learning.rnn_adapter import RNNAdapter, cell_trace, demo_cell, gradient_experiment, variable_length_demo, masked_mse
from learning.rnn_training import RNNTrainingConfig, sequence_splits, train_sequences, evaluate_sequence_test


@pytest.mark.parametrize('activation',['tanh','relu'])
def test_cell_independent_reference_and_recurrence(activation):
    x,wi,wh,bi,bh = demo_cell()
    bi = bi+.1; bh = bh-.03; h0 = torch.full((3,),.2)
    states,steps = cell_trace(x,wi,wh,bi,bh,h0,activation)
    ref = torch.nn.RNNCell(2,3,nonlinearity=activation)
    with torch.no_grad():
        ref.weight_ih.copy_(wi);ref.weight_hh.copy_(wh);ref.bias_ih.copy_(bi);ref.bias_hh.copy_(bh)
    h = h0
    for t in range(len(x)):
        h = ref(x[t],h)
        torch.testing.assert_close(states[t],h)
        torch.testing.assert_close(steps[t]['preactivation'],wi@x[t]+bi+wh@steps[t]['previous']+bh)
    reverse,_ = cell_trace(x.flip(0),wi,wh,bi,bh,h0,activation)
    assert not torch.allclose(states[-1],reverse[-1])


@pytest.mark.parametrize('layers',[1,3])
@pytest.mark.parametrize('batch_first',[True,False])
@pytest.mark.parametrize('selection',['final','sequence'])
@pytest.mark.parametrize('activation',['tanh','relu'])
@pytest.mark.parametrize('bias',[True,False])
def test_shapes_count_trace_and_torch_parity(layers,batch_first,selection,activation,bias):
    spec = RNNModelSpec(2,4,layers,3,activation,selection,bias,batch_first)
    assert RNNModelSpec.from_dict(json.loads(json.dumps(spec.as_dict()))) == spec
    x = torch.randn((2,5,2) if batch_first else (5,2,2))
    model,trace = RNNAdapter().trace(spec,x)
    assert sum(p.numel() for p in model.parameters()) == spec.parameter_count
    rows = spec.infer_shapes(2,5)
    assert tuple(model(x).shape) == rows[-1]['shape']
    torch.testing.assert_close(trace[-1]['tensor'],model(x))
    independent = torch.nn.RNN(2,4,layers,nonlinearity=activation,bias=bias,batch_first=batch_first)
    independent.load_state_dict(model.rnn.state_dict())
    seq,hn = independent(x)
    assert tuple(hn.shape) == (layers,2,4)
    torch.testing.assert_close(model(x),model.head(hn[-1] if selection=='final' else seq))
    if layers == 1:
        xb = x if batch_first else x.transpose(0,1)
        states,_ = cell_trace(xb[0],independent.weight_ih_l0,independent.weight_hh_l0,
                              independent.bias_ih_l0 if bias else None,independent.bias_hh_l0 if bias else None,activation=activation)
        torch.testing.assert_close(states,(seq if batch_first else seq.transpose(0,1))[0])


def test_neutral_and_legacy_specs_progress():
    import learning.rnn_spec as module
    from learning.spec import ModelSpec
    from learning.cnn_spec import digits_spec, CNNModelSpec
    from learning.progress import empty_progress, load_progress, export_progress, mark_complete, record_quiz
    assert 'import torch' not in inspect.getsource(module)
    dense = ModelSpec.for_lesson('feed_forward',2,(3,),1)
    assert ModelSpec.from_dict(dense.as_dict()) == dense
    cnn = digits_spec(); assert CNNModelSpec.from_dict(cnn.as_dict()) == cnn
    old = load_progress(export_progress(mark_complete(empty_progress(),'cnn')))
    new = record_quiz(mark_complete(old,'rnn'),'rnn',10,12)
    assert load_progress(export_progress(new)) == new and 'cnn' in new['completed']
    with pytest.raises(ValueError): RNNModelSpec(hidden_size=0)
    with pytest.raises(ValueError): RNNModelSpec(activation='sigmoid')
    with pytest.raises(ValueError): RNNModelSpec.from_dict(dense.as_dict())


def test_bptt_clipping_truncation_and_update():
    full = gradient_experiment(12,1.3,'relu',clip=.2)
    tiny = gradient_experiment(12,.3,'relu')
    assert full['state_gradients'][0] > full['state_gradients'][-1]
    assert tiny['state_gradients'][0] < 1e-5
    assert full['norm_before'] > .2 and full['norm_after'] <= .200001
    torch.testing.assert_close(full['after'],full['before']-.01*full['gradient_after'])
    # Independent derivative of terminal positive ReLU state w.r.t. every input.
    np.testing.assert_allclose(full['input_gradients'].flatten(),[1.3**i for i in range(11,-1,-1)])
    cut = gradient_experiment(12,1.3,'relu',truncate=4)
    assert cut['boundaries'] == [4,8]
    assert cut['input_gradients'][:8].count_nonzero() == 0
    assert all(v==0 for v in cut['state_gradients'][:8])
    assert cut['input_gradients'][8:].count_nonzero() == 4
    assert cut['gradient_before'].abs().item() > 0
    assert sum(full['weight_contributions']) == pytest.approx(full['gradient_before'].item())
    assert sum(cut['weight_contributions']) == pytest.approx(cut['gradient_before'].item())


def test_variable_lengths_individual_runs_and_masked_loss():
    result = variable_length_demo()
    torch.testing.assert_close(result['final_valid'],result['packed_hidden'])
    assert not torch.allclose(result['wrong_padded_last'][1],result['final_valid'][1])
    model = RNNAdapter().build(RNNModelSpec(hidden_size=3))
    for i,length in enumerate(result['lengths']):
        _,hidden = model.encode(result['input'][i:i+1,:length])
        torch.testing.assert_close(hidden[-1,0],result['final_valid'][i])
    poisoned = result['input'].clone(); poisoned[~result['mask']] = 1e4
    torch.testing.assert_close(model(poisoned,result['lengths']),model(result['input'],result['lengths']))
    p = torch.ones(3,5,1,requires_grad=True); target = torch.zeros_like(p)
    target[~result['mask']] = float('nan')
    loss = masked_mse(p,target,result['lengths']);loss.backward()
    assert loss.item()==1 and p.grad[~result['mask']].count_nonzero()==0


@pytest.mark.parametrize('task',['classification','forecasting','memory'])
def test_dataset_reproducibility_isolation_and_order(task):
    a = sequence_splits(task);b = sequence_splits(task)
    for key in ('train','validation','test'):
        torch.testing.assert_close(a[key][0],b[key][0]);torch.testing.assert_close(a[key][1],b[key][1])
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(a['indices'][left]) & set(a['indices'][right])
        assert not set(a['source_series'][left]) & set(a['source_series'][right])
    x,y = a['train']
    if task=='classification':
        torch.testing.assert_close(y,(x[:,0,0]<x[:,-1,0]).long())
        assert torch.equal(1-y,(x.flip(1)[:,0,0]<x.flip(1)[:,-1,0]).long())
    if task=='memory': assert torch.equal(y,(x[:,0,0]>0).long())


@pytest.mark.parametrize('task',['classification','forecasting'])
@pytest.mark.parametrize('optimizer',['SGD','Adam'])
@pytest.mark.parametrize('layers',[1,2])
def test_training_and_no_heldout_leakage(task,optimizer,layers):
    cfg = RNNTrainingConfig(task=task,samples=100,epochs=3,optimizer=optimizer,horizon=2,num_layers=layers)
    data = sequence_splits(task,100,horizon=2)
    run = train_sequences(cfg,data)
    poisoned = dict(data)
    for key in ('validation','test'):
        x,y = data[key]; poisoned[key] = (x*0+10,y.flip(0))
    other = train_sequences(cfg,poisoned)
    for a,b in zip(run['model'].parameters(),other['model'].parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(math.isfinite(row['train_loss']) for row in run['history'])
    assert any(v>0 for v in run['parameter_changes'].values())
    assert all(row['clipped_norm'] <= cfg.clip+1e-5 for row in run['history'])
    result = evaluate_sequence_test(run)
    assert math.isfinite(result['loss']) and result['prediction'].shape[0] == 20


@pytest.mark.parametrize('task',['classification','forecasting'])
def test_learns_small_controlled_task(task):
    run = train_sequences(RNNTrainingConfig(task=task,samples=240,epochs=16,hidden_size=16))
    assert run['history'][-1]['train_loss'] < run['history'][0]['train_loss']
    result = evaluate_sequence_test(run)
    assert result['metric'] < .3 if task=='forecasting' else result['metric'] > .7
