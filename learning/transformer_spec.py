"""Small, standard equal-head Transformer laboratory configuration."""
from dataclasses import dataclass
import math
from .embedding_spec import positive_int, EmbeddingModelSpec


@dataclass(frozen=True)
class TransformerSpec:
    d_model: int = 8
    num_heads: int = 2
    d_ff: int = 16
    activation: str = 'GELU'
    norm: str = 'Pre-LayerNorm'
    epsilon: float = 1e-5
    seed: int = 42

    def __post_init__(self):
        for name, limit in (('d_model', 32), ('num_heads', 8), ('d_ff', 128)):
            positive_int(getattr(self, name), name)
            if getattr(self, name) > limit:
                raise ValueError(f'{name} must be at most {limit}')
        if self.d_model % self.num_heads:
            raise ValueError('d_model must be divisible by num_heads in standard mode')
        if self.activation not in ('ReLU', 'GELU'):
            raise ValueError('Choose ReLU or GELU')
        if self.norm not in ('Pre-LayerNorm', 'Post-LayerNorm'):
            raise ValueError('Choose Pre-LayerNorm or Post-LayerNorm')
        if isinstance(self.epsilon, bool) or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('LayerNorm epsilon must be finite and positive')
        EmbeddingModelSpec(seed=self.seed)

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads
