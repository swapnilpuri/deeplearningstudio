"""Validated CPU laboratory dimensions; independent of presentation/framework layers."""
from dataclasses import dataclass
from .embedding_spec import positive_int, EmbeddingModelSpec


@dataclass(frozen=True)
class AttentionSpec:
    embedding_dim: int = 4
    key_dim: int = 3
    value_dim: int = 2
    seed: int = 42
    initialization: str = 'normal'

    def __post_init__(self):
        for name in ('embedding_dim', 'key_dim', 'value_dim'):
            positive_int(getattr(self, name), name)
            if getattr(self, name) > 32:
                raise ValueError('Laboratory dimensions must be at most 32')
        EmbeddingModelSpec(seed=self.seed)
        if self.initialization not in ('normal', 'identity', 'zeros'):
            raise ValueError('Choose normal, identity or zeros initialization')
