"""Reference PyTorch backend: all demonstrations execute actual tensor operations."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .spec import ModelSpec

ACTIVATIONS = {'linear': nn.Identity, 'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid,
               'Tanh': nn.Tanh, 'GELU': nn.GELU}

@dataclass
class StepResult:
    name: str
    shape: tuple[int, ...]
    values: list
    explanation: str

class PyTorchAdapter:
    framework = 'pytorch'

    def build(self, spec: ModelSpec) -> nn.Sequential:
        if spec.framework != self.framework:
            raise ValueError(f'{spec.framework} backend not implemented; use pytorch')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            layers = []
            previous = spec.input_features
            for layer in spec.layers:
                layers.append(nn.Linear(previous, layer.units, bias=spec.bias))
                if layer.activation != 'linear':
                    layers.append(ACTIVATIONS[layer.activation]())
                previous = layer.units
            return nn.Sequential(*layers)

    def trace(self, spec: ModelSpec, values: list[float]) -> tuple[nn.Sequential, list[StepResult]]:
        if len(values) != spec.input_features or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f'Enter {spec.input_features} finite input numbers')
        model = self.build(spec).eval()
        x = torch.tensor([values], dtype=torch.float32)
        steps = [StepResult('Input', tuple(x.shape), x.tolist(), 'Input batch: one sample × feature count.')]
        with torch.no_grad():
            for i, module in enumerate(model):
                x = module(x)
                description = ('z = x @ W.T + b (linear transformation)' if isinstance(module, nn.Linear)
                               else f'a = {type(module).__name__}(z) (nonlinear activation)')
                steps.append(StepResult(f'{i+1}. {type(module).__name__}', tuple(x.shape), x.tolist(), description))
        return model, steps

    def gradient_step(self, spec: ModelSpec, values: list[float], target: list[float],
                      learning_rate: float = 0.1) -> dict:
        if len(target) != spec.layers[-1].units or not all(math.isfinite(float(v)) for v in target):
            raise ValueError(f'Target needs {spec.layers[-1].units} finite numbers')
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError('Learning rate must be in (0, 1]')
        if len(values) != spec.input_features or not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Invalid input vector')
        model = self.build(spec).train()
        x = torch.tensor([values], dtype=torch.float32)
        y = torch.tensor([target], dtype=torch.float32)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        prediction = model(x)
        loss = nn.MSELoss()(prediction, y)
        loss.backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        optimizer.step()
        updates = []
        for name, p in model.named_parameters():
            old, gradient = before[name], gradients[name]
            updates.append({'parameter': name, 'shape': list(p.shape),
                            'before': old.flatten().tolist(), 'gradient': gradient.flatten().tolist(),
                            'after': p.detach().flatten().tolist(),
                            'gradient_norm': gradient.norm().item(),
                            'max_update': (p.detach() - old).abs().max().item()})
        new_prediction = model(x).detach().tolist()
        return {'prediction_before': prediction.detach().tolist(), 'target': y.tolist(),
                'loss_before': loss.item(), 'prediction_after': new_prediction,
                'loss_after': nn.MSELoss()(model(x), y).detach().item(),
                'learning_rate': learning_rate, 'parameters': updates}

    def parameter_count(self, spec: ModelSpec) -> int:
        return sum(p.numel() for p in self.build(spec).parameters())

# Explicit registry contract. TensorFlow adapter can be registered later without modifying ModelSpec.
ADAPTERS = {'pytorch': PyTorchAdapter}

def get_adapter(framework: str = 'pytorch'):
    if framework not in ADAPTERS:
        raise NotImplementedError(f'{framework} adapter is not installed. Available: {list(ADAPTERS)}')
    return ADAPTERS[framework]()
