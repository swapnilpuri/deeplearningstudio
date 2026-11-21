import json
import math
import pytest
import torch
from learning.spec import LayerSpec, ModelSpec
from learning.pytorch_adapter import PyTorchAdapter, get_adapter
from learning.progress import empty_progress, mark_complete, record_quiz, export_progress, load_progress
from learning.content import LESSONS
from visualization.learning_graphs import tensor_heatmap, gradient_bar

@pytest.fixture
def spec():
    return ModelSpec.for_lesson('feed_forward', 2, (3,), 1, 'Tanh', True, 42)

def test_spec_round_trip_no_framework_dependency(spec):
    assert ModelSpec.from_dict(json.loads(json.dumps(spec.as_dict()))) == spec
    assert spec.sizes == (2,3,1)
    with pytest.raises(ValueError):
        LayerSpec(0)
    with pytest.raises(ValueError):
        ModelSpec.for_lesson('feed_forward', 2, (3,4), 1)
    with pytest.raises(ValueError):
        ModelSpec.for_lesson('deep_neural_network', 2, (3,), 1)
    with pytest.raises(NotImplementedError):
        get_adapter('tensorflow')

def test_pytorch_matches_legacy_model(spec):
    from models.feed_forward import build_model
    model = PyTorchAdapter().build(spec)
    legacy = build_model(spec.sizes, 'Tanh', True, 42)
    for a, b in zip(model.parameters(), legacy.parameters()):
        assert torch.equal(a, b)
    assert PyTorchAdapter().parameter_count(spec) == 13

def test_forward_trace_and_shaped_tensors(spec):
    adapter = PyTorchAdapter()
    model, steps = adapter.trace(spec, [1, -1])
    assert [s.shape for s in steps] == [(1,2),(1,3),(1,3),(1,1)]
    assert torch.allclose(torch.tensor(steps[-1].values), model(torch.tensor([[1.,-1.]])))
    with pytest.raises(ValueError):
        adapter.trace(spec, [float('nan'), 1])

def test_autograd_sgd_and_mse(spec):
    data = PyTorchAdapter().gradient_step(spec, [1,2], [0], .1)
    assert math.isfinite(data['loss_before'])
    assert data['loss_after'] <= data['loss_before'] + 1e-6
    assert any(item['gradient_norm'] > 0 for item in data['parameters'])
    for item in data['parameters']:
        for before, grad, after in zip(item['before'], item['gradient'], item['after']):
            assert after == pytest.approx(before - .1 * grad, abs=1e-6)
    with pytest.raises(ValueError):
        PyTorchAdapter().gradient_step(spec, [1,2], [0,1])

def test_deep_spec_and_gradients():
    spec = ModelSpec.for_lesson('deep_neural_network', 3, (5,4), 1, 'ReLU', True, 3)
    result = PyTorchAdapter().gradient_step(spec, [1,2,3], [1])
    assert len(result['parameters']) == 6
    assert len(PyTorchAdapter().trace(spec,[1,2,3])[1]) == 6

def test_progress_roundtrip():
    p = mark_complete(empty_progress(),'feed_forward')
    p = record_quiz(p, 'feed_forward', 2, 3)
    assert load_progress(export_progress(p)) == p
    with pytest.raises(ValueError):
        load_progress('{"schema_version": 999}')
    with pytest.raises(ValueError):
        record_quiz(p, 'feed_forward', 4, 3)

def test_lesson_quizzes_and_visuals():
    for name, content in LESSONS.items():
        assert content['questions']
        assert all(0 <= q[2] < len(q[1]) for q in content['questions'])
    assert len(tensor_heatmap([[1,2]]).data) == 1
    assert len(gradient_bar([{'parameter':'a','gradient_norm':.2}]).data) == 1

def test_workspace_hook_preserves_v3():
    from pathlib import Path
    app = (
    Path(__file__).resolve()
    .parents[1]
    .joinpath('app.py')
    .read_text(encoding='utf-8')
    )
    for name in ('Neural Network Studio','Training Studio','Dataset Management','Learning Studio'):
        assert name in app
    compile(app, 'app.py', 'exec')
