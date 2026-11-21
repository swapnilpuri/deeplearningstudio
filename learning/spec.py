"""Immutable, framework-independent network definitions. No torch imports."""
from dataclasses import dataclass
from typing import Literal

Activation = Literal['ReLU', 'Sigmoid', 'Tanh', 'GELU']

@dataclass(frozen=True)
class LayerSpec:
    units: int
    activation: str = 'linear'

    def __post_init__(self):
        if not isinstance(self.units, int) or isinstance(self.units, bool) or not 1 <= self.units <= 128:
            raise ValueError('Layer width must be an integer between 1 and 128')
        if self.activation not in ('linear', 'ReLU', 'Sigmoid', 'Tanh', 'GELU'):
            raise ValueError('Unsupported activation: ' + str(self.activation))

@dataclass(frozen=True)
class ModelSpec:
    architecture: Literal['feed_forward', 'deep_neural_network']
    input_features: int
    layers: tuple[LayerSpec, ...]
    bias: bool = True
    seed: int = 42
    framework: str = 'pytorch'

    def __post_init__(self):
        if self.architecture not in ('feed_forward', 'deep_neural_network'):
            raise ValueError('Unknown architecture')
        if not isinstance(self.input_features, int) or isinstance(self.input_features, bool) or not 1 <= self.input_features <= 128:
            raise ValueError('input_features must be 1..128')
        if not self.layers or len(self.layers) > 9:
            raise ValueError('Specify 1..9 output/hidden layers')
        if self.layers[-1].activation != 'linear':
            raise ValueError('The output layer must have linear activation (pre-loss logits)')
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0 or self.seed > 2147483647:
            raise ValueError('Invalid random seed')
        if self.architecture == 'feed_forward' and len(self.layers) > 2:
            raise ValueError('Feed-forward lesson supports zero or one hidden layer')
        if self.architecture == 'deep_neural_network' and len(self.layers) < 3:
            raise ValueError('Deep neural network requires at least two hidden layers')

    @property
    def sizes(self) -> tuple[int, ...]:
        return (self.input_features, *(l.units for l in self.layers))

    def as_dict(self) -> dict:
        return {'architecture': self.architecture, 'input_features': self.input_features,
                'layers': [{'units': l.units, 'activation': l.activation} for l in self.layers],
                'bias': self.bias, 'seed': self.seed, 'framework': self.framework}

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelSpec':
        return cls(architecture=data['architecture'], input_features=data['input_features'],
                   layers=tuple(LayerSpec(**l) for l in data['layers']),
                   bias=data.get('bias', True), seed=data.get('seed', 42),
                   framework=data.get('framework', 'pytorch'))

    @classmethod
    def for_lesson(cls, architecture: str, inputs: int, hidden: tuple[int, ...], outputs: int,
                   activation: str = 'Tanh', bias: bool = True, seed: int = 42):
        return cls(architecture, inputs,
                   tuple(LayerSpec(n, activation) for n in hidden) + (LayerSpec(outputs),), bias, seed)
