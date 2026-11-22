"""Validated CPU-sized encoder-decoder configurations (batch-first throughout)."""
from dataclasses import dataclass, field
import math
from .embedding_spec import positive_int
from .transformer_spec import TransformerSpec


@dataclass(frozen=True)
class Seq2SeqSpec:
    block: TransformerSpec = field(default_factory=lambda: TransformerSpec(d_model=16, d_ff=32))
    source_vocabulary: int = 7
    target_vocabulary: int = 7
    encoder_layers: int = 1
    decoder_layers: int = 1
    positional: str = 'sinusoidal'
    dropout: float = 0.0
    max_length: int = 12

    def __post_init__(self):
        for name, low, high in (('source_vocabulary', 4, 64), ('target_vocabulary', 4, 64),
                                ('encoder_layers', 1, 4), ('decoder_layers', 1, 4),
                                ('max_length', 3, 24)):
            value = getattr(self, name)
            positive_int(value, name)
            if not low <= value <= high:
                raise ValueError(f'{name} must be in [{low}, {high}]')
        if self.positional not in ('sinusoidal', 'learned'):
            raise ValueError('Choose sinusoidal or learned positions')
        if isinstance(self.dropout, bool) or not math.isfinite(self.dropout) or not 0 <= self.dropout <= .5:
            raise ValueError('Dropout must be finite in [0, 0.5]')
