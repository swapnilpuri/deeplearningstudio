import csv
import io
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from learning.cnn_adapter import CNNAdapter, convolution_explorer, convolution_gradient_check, single_sgd_step
from learning.cnn_spec import CNNLayerSpec, CNNModelSpec, digits_spec
from learning.cnn_training import CNNTrainingConfig, digits_splits, train_digits
from learning.progress import load_progress, export_progress, mark_complete, record_quiz
from training.exports import history_csv


def test_history_csv_retains_later_validation_gradient_and_future_metrics():
    records = [{'epoch':0,'loss':1}, {'epoch':1,'loss':.5,'val_loss':.6,'val_score':.8,'grad_norm':.12,'future_metric':3}]
    rows = list(csv.DictReader(io.StringIO(history_csv(records))))
    assert len(rows) == 2
    assert rows[0]['grad_norm'] == ''
    for key in ('val_loss','val_score','grad_norm','future_metric'):
        assert float(rows[1][key]) == records[1][key]
    assert history_csv([]) == ''


def test_csv_export_from_real_training_history():
    from training.datasets import built_in_dataset, prepare_dataset
    from training.engine import Experiment, ExperimentConfig
    frame,target,task = built_in_dataset('Iris')
    bundle = prepare_dataset(frame,name='Iris',target=target,task=task)
    exp = Experiment(ExperimentConfig(dataset='Iris'),bundle)
    exp.train_epochs(1)
    rows = list(csv.DictReader(io.StringIO(history_csv(exp.history))))
    assert len(rows) == 2
    for key in ('val_loss','val_score','grad_norm'):
        assert float(rows[-1][key]) == pytest.approx(exp.history[-1][key])


def test_m4_progress_survives_cnn_edits_and_export():
    old = {'schema_version':1,'completed':['feed_forward','deep_neural_network'],
           'quiz_scores':{'feed_forward':{'correct':2,'total':3,'recorded_utc':'2026-01-01T00:00:00Z'}},
           'exercise_notes':{'feed_forward':'Original notes — café'}}
    progress = load_progress(json.dumps(old,ensure_ascii=False).encode('utf-8'))
    progress = mark_complete(progress,'cnn')
    progress = record_quiz(progress,'cnn',10,12)
    restored = load_progress(export_progress(progress))
    assert restored['exercise_notes'] == old['exercise_notes']
    assert restored['quiz_scores']['feed_forward'] == old['quiz_scores']['feed_forward']
    assert set(restored['completed']) == set(old['completed']) | {'cnn'}


@pytest.mark.parametrize('dilation',[(2,1),(1,3),(2,2)])
def test_rectangular_dilated_rgb_trace(dilation):
    g = torch.Generator().manual_seed(8)
    x = torch.randn(3,7,9,generator=g)
    w = torch.randn(2,3,2,3,generator=g)
    b = torch.tensor([.5,-.75])
    result = convolution_explorer(x,w,b,stride=(2,1),padding=(1,2),row=1,col=2,filter_index=1,dilation=dilation)
    expected = F.conv2d(x[None],w,b,stride=(2,1),padding=(1,2),dilation=dilation)[0]
    torch.testing.assert_close(result['output'],expected)
    assert result['sum_products']+result['bias'] == pytest.approx(expected[1,1,2].item(),abs=2e-5)
    assert len(result['sample_positions']) == 6


def test_rectangular_model_hidden_head_bias_shapes_and_trace():
    spec = CNNModelSpec(3,8,12,(
        CNNLayerSpec('conv2d',out_channels=4,kernel_size=(2,3),dilation=(2,1),bias=False),
        CNNLayerSpec('relu'),CNNLayerSpec('avgpool2d',kernel_size=(2,2),stride=(2,2)),
        CNNLayerSpec('flatten'),CNNLayerSpec('linear',out_features=16),
        CNNLayerSpec('relu'),CNNLayerSpec('linear',out_features=3,bias=False)))
    model,trace = CNNAdapter().trace(spec)
    rows = spec.infer_shapes()
    assert [r['shape'] for r in rows] == [t['shape'] for t in trace]
    assert rows[4]['shape'] == (1,60)
    assert rows[5]['weights'] == (16,60) and rows[5]['bias'] == (16,)
    assert rows[1]['bias'] is None and rows[-1]['bias'] is None
    assert sum(r['parameters'] for r in rows) == sum(p.numel() for p in model.parameters())


def test_deterministic_gradient_check_and_adam_step():
    assert convolution_gradient_check()
    spec = digits_spec(filters1=2,filters2=2)
    x = torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)/63
    r = single_sgd_step(spec,x,torch.tensor([3]),.001,'Adam')
    assert np.isfinite(r['loss_after'])
    p = r['parameters']['0.weight']
    expected = p['before']-.001*p['gradient']/(p['gradient'].abs()+1e-8)
    torch.testing.assert_close(p['after'],expected)


@pytest.mark.parametrize('field,value',[('input_channels',True),('image_height',8.5),('image_width',False)])
def test_invalid_noninteger_dimensions(field,value):
    with pytest.raises(ValueError):
        replace(digits_spec(),**{field:value})


@pytest.mark.parametrize('optimizer',['Adam','SGD'])
def test_training_reproducibility_and_no_heldout_updates(monkeypatch,optimizer):
    import learning.cnn_training as module
    config = CNNTrainingConfig(epochs=1,batch_size=128,optimizer=optimizer,filters1=2,filters2=2)
    baseline = train_digits(config)
    original = digits_splits
    def changed_heldout(seed):
        split = original(seed)
        # Poison held-out values; trained weights must remain exactly identical.
        for part in ('validation','test'):
            x,y = split[part]
            split[part] = (torch.zeros_like(x), (y+1)%10)
        return split
    monkeypatch.setattr(module,'digits_splits',changed_heldout)
    poisoned = train_digits(config)
    for a,b in zip(baseline['model'].parameters(),poisoned['model'].parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert baseline['history'][0]['train_loss'] == poisoned['history'][0]['train_loss']


def test_digits_normalization_uses_fixed_domain_divisor():
    from sklearn.datasets import load_digits
    raw = load_digits()
    split = digits_splits(7)
    again = digits_splits(7)
    for part,indices in split['indices'].items():
        np.testing.assert_array_equal(indices,again['indices'][part])
        np.testing.assert_allclose(split[part][0][:,0].numpy(),raw.images[indices]/16,rtol=0,atol=0)


def test_legacy_onnx_export_and_checker(tmp_path):
    import onnx
    from models.feed_forward import build_model
    model = build_model([4,6,2])
    path = tmp_path/'network.onnx'
    torch.onnx.export(model,(torch.zeros(1,4),),str(path),input_names=['input'],output_names=['output'],dynamo=True,external_data=False)
    exported = onnx.load(path)
    onnx.checker.check_model(exported)
    assert len(exported.graph.initializer) == len(list(model.parameters()))
