import math
import pytest
import torch
from training.engine import Experiment, ExperimentConfig, DATASETS, is_classification
from visualization.training_graphs import dataset_figure, decision_figure, history_figure

@pytest.mark.parametrize('dataset', DATASETS)
def test_shapes_and_reproducibility(dataset):
    config = ExperimentConfig(dataset=dataset, samples=100, seed=88)
    a, b = Experiment(config), Experiment(config)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.y, b.y)
    assert a.x.shape == (100, 2 if is_classification(dataset) else 1)
    assert a.y.shape == ((100,) if is_classification(dataset) else (100, 1))
    assert a.model(a.x).shape == (100, 2 if is_classification(dataset) else 1)
    assert a.epoch == 0 and len(a.history) == 1

@pytest.mark.parametrize('name', ('SGD', 'Adam', 'RMSprop'))
def test_optimizer_and_gradients(name):
    exp = Experiment(ExperimentConfig(dataset='Linear regression', samples=120, optimizer=name, lr=.02, seed=9))
    initial = exp.history[0]['loss']
    exp.train_epochs(45)
    assert exp.epoch == 45
    assert len(exp.history) == 46
    assert exp.history[-1]['loss'] < initial
    assert len(exp.last_gradients) > 0
    assert all(math.isfinite(item['grad_norm']) for item in exp.last_gradients)


def test_classification_learns_and_resets():
    exp = Experiment(ExperimentConfig(dataset='Two moons', hidden=(16,16), samples=200, lr=.025))
    before = exp.history[0]['loss']
    exp.train_epochs(35)
    assert exp.history[-1]['loss'] < before
    assert exp.history[-1]['score'] > .8
    assert len(decision_figure(exp).data) == 3
    assert len(history_figure(exp, 'loss').data) == 1
    exp.reset()
    assert exp.epoch == 0 and len(exp.history) == 1


def test_regression_chart_and_updates():
    exp = Experiment(ExperimentConfig(dataset='Sine regression'))
    assert len(dataset_figure(exp).data) == 2
    exp.train_epochs(1)
    assert exp.history[1]['grad_norm'] > 0


def test_invalid_values():
    with pytest.raises(ValueError):
        Experiment(ExperimentConfig(dataset='unknown'))
    with pytest.raises(ValueError):
        Experiment(ExperimentConfig(lr=0))
    with pytest.raises(ValueError):
        Experiment(ExperimentConfig(samples=3))
    with pytest.raises(ValueError):
        Experiment(ExperimentConfig()).train_epochs(0)
