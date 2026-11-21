"""CPU-first reproducible experiments for regression and binary classification."""
from dataclasses import dataclass
import copy
import math

import numpy as np
import torch
from torch import nn

from models.feed_forward import build_model


DATASETS = ('Linear regression', 'Sine regression', 'Two moons', 'Concentric circles')
OPTIMIZERS = ('SGD', 'Adam', 'RMSprop')


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = 'Two moons'
    hidden: tuple[int, ...] = (16, 16)
    activation: str = 'Tanh'
    bias: bool = True
    seed: int = 42
    samples: int = 320
    noise: float = 0.12
    optimizer: str = 'Adam'
    lr: float = 0.02
    batch_size: int = 32
    momentum: float = 0.9
    weight_decay: float = 0.0


def is_classification(dataset: str) -> bool:
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    return dataset in ('Two moons', 'Concentric circles')


def make_dataset(cfg: ExperimentConfig):
    """Return (x: float32[n, features], y: float32[n,1] or int64[n])."""
    if cfg.samples < 20:
        raise ValueError('At least 20 samples are required')
    if not 0 <= cfg.noise <= 1:
        raise ValueError('Noise must be between 0 and 1')
    rng = np.random.default_rng(cfg.seed)
    n = cfg.samples
    if cfg.dataset == 'Linear regression':
        x = rng.uniform(-2, 2, (n, 1)).astype('float32')
        y = (1.8 * x - 0.7 + rng.normal(0, cfg.noise, (n, 1))).astype('float32')
    elif cfg.dataset == 'Sine regression':
        x = rng.uniform(-3.14, 3.14, (n, 1)).astype('float32')
        y = (np.sin(2 * x) + rng.normal(0, cfg.noise, (n, 1))).astype('float32')
    elif cfg.dataset == 'Two moons':
        outer = (n + 1) // 2
        inner = n - outer
        t1 = rng.uniform(0, np.pi, outer)
        t2 = rng.uniform(0, np.pi, inner)
        a = np.column_stack((np.cos(t1), np.sin(t1)))
        b = np.column_stack((1 - np.cos(t2), 0.5 - np.sin(t2)))
        x = np.vstack((a, b)) + rng.normal(0, cfg.noise, (n, 2))
        y = np.concatenate((np.zeros(outer), np.ones(inner)))
    elif cfg.dataset == 'Concentric circles':
        outer = (n + 1) // 2
        inner = n - outer
        t1 = rng.uniform(0, 2*np.pi, outer)
        t2 = rng.uniform(0, 2*np.pi, inner)
        a = np.column_stack((np.cos(t1), np.sin(t1)))
        b = 0.45 * np.column_stack((np.cos(t2), np.sin(t2)))
        x = np.vstack((a, b)) + rng.normal(0, cfg.noise, (n, 2))
        y = np.concatenate((np.zeros(outer), np.ones(inner)))
    else:
        raise ValueError(f'Unknown dataset: {cfg.dataset}')
    if is_classification(cfg.dataset):
        perm = rng.permutation(n)
        return torch.tensor(x[perm], dtype=torch.float32), torch.tensor(y[perm], dtype=torch.long)
    perm = rng.permutation(n)
    return torch.tensor(x[perm], dtype=torch.float32), torch.tensor(y[perm], dtype=torch.float32)


class Experiment:
    def __init__(self, cfg: ExperimentConfig, bundle=None):
        if cfg.optimizer not in OPTIMIZERS:
            raise ValueError('Invalid optimizer')
        if cfg.lr <= 0 or not math.isfinite(cfg.lr):
            raise ValueError('Learning rate must be positive and finite')
        if cfg.batch_size < 1:
            raise ValueError('Batch size must be positive')
        self.cfg = cfg
        self.bundle = bundle
        if bundle is None:
            self.x, self.y = make_dataset(cfg)
            self.val_x = self.val_y = self.test_x = self.test_y = None
            self.classification = is_classification(cfg.dataset)
            out_features = 2 if self.classification else 1
        else:
            self.x, self.y = bundle.train_x, bundle.train_y
            self.val_x, self.val_y = bundle.val_x, bundle.val_y
            self.test_x, self.test_y = bundle.test_x, bundle.test_y
            self.classification = bundle.classification
            out_features = bundle.output_size
        self.sizes = (self.x.shape[1], *cfg.hidden, out_features)
        self.model = build_model(self.sizes, cfg.activation, cfg.bias, cfg.seed)
        self.model.train()
        self.loss_fn = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()
        if cfg.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history = []
        self.epoch = 0
        self.last_gradients = []
        self._generator = torch.Generator().manual_seed(cfg.seed + 12345)
        self.initial_parameters = copy.deepcopy(self.model.state_dict())
        self.record_metrics(grad_norm=0.0)

    @torch.no_grad()
    def record_metrics(self, grad_norm: float):
        self.model.eval()
        logits = self.model(self.x)
        loss = float(self.loss_fn(logits, self.y).item())
        if self.classification:
            score = float((logits.argmax(dim=1) == self.y).float().mean().item())
        else:
            score = float(torch.sqrt(self.loss_fn(logits, self.y)).item())
        metrics = {'epoch': self.epoch, 'loss': loss, 'score': score, 'grad_norm': grad_norm}
        if self.val_x is not None:
            val_logits = self.model(self.val_x)
            metrics['val_loss'] = float(self.loss_fn(val_logits, self.val_y).item())
            metrics['val_score'] = float((val_logits.argmax(dim=1) == self.val_y).float().mean().item()) if self.classification else float(torch.sqrt(self.loss_fn(val_logits, self.val_y)).item())
        self.history.append(metrics)
        self.model.train()
        return self.history[-1]

    def train_epochs(self, count: int = 1):
        """Train additional epochs. Gradient norms are measured before optimizer.step()."""
        if not 1 <= count <= 500:
            raise ValueError('Train between 1 and 500 epochs per run')
        for _ in range(count):
            indices = torch.randperm(len(self.x), generator=self._generator)
            grad_norm_sum = 0.0
            grad_samples = 0
            for start in range(0, len(indices), self.cfg.batch_size):
                index = indices[start:start+self.cfg.batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(self.x[index])
                loss = self.loss_fn(prediction, self.y[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss. Reset experiment and lower the learning rate.')
                loss.backward()
                batch_gradients = []
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        norm = float(p.grad.detach().norm().item())
                        batch_gradients.append({'parameter': name, 'grad_norm': norm})
                norm_sq = sum(item['grad_norm']**2 for item in batch_gradients)
                grad_norm_sum += math.sqrt(norm_sq) * len(index)
                grad_samples += len(index)
                self.last_gradients = batch_gradients
                self.optimizer.step()
            self.epoch += 1
            self.record_metrics(grad_norm=grad_norm_sum / max(1, grad_samples))
        return self.history[-1]

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        output = self.model(x)
        self.model.train()
        return output

    def reset(self):
        self.__init__(self.cfg, self.bundle)

    @torch.no_grad()
    def test_metrics(self):
        """Evaluate the held-out test set, which is never used for optimization."""
        if self.test_x is None:
            return None
        previous = self.model.training
        self.model.eval()
        logits = self.model(self.test_x)
        loss = float(self.loss_fn(logits, self.test_y).item())
        score = float((logits.argmax(dim=1) == self.test_y).float().mean().item()) if self.classification else float(torch.sqrt(self.loss_fn(logits, self.test_y)).item())
        self.model.train(previous)
        return {'loss': loss, 'score': score, 'samples': len(self.test_x)}
