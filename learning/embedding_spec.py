"""Version 5 embedding classifier definition; deliberately framework neutral."""
from dataclasses import asdict, dataclass


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class EmbeddingModelSpec:
    vocabulary_size: int = 12
    embedding_dim: int = 4
    output_size: int = 2
    padding_idx: int | None = 0
    pooling: str = 'mean'
    positional: str = 'none'
    max_length: int = 20
    seed: int = 42
    schema_version: int = 5
    architecture: str = 'embedding'

    def __post_init__(self):
        for name in ('vocabulary_size', 'embedding_dim', 'output_size', 'max_length'):
            positive_int(getattr(self, name), name)
        if self.padding_idx is not None and (type(self.padding_idx) is not int or not 0 <= self.padding_idx < self.vocabulary_size):
            raise ValueError('Padding index must be inside the vocabulary')
        if self.pooling not in ('mean', 'sum', 'max') or self.positional not in ('none', 'learned', 'sinusoidal'):
            raise ValueError('Unsupported pooling or positional representation')
        if type(self.seed) is not int or not 0 <= self.seed <= 2147483647:
            raise ValueError('Seed must be an integer in [0,2147483647]')
        if type(self.schema_version) is not int or self.schema_version != 5 or self.architecture != 'embedding':
            raise ValueError('Expected embedding schema v5')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != 5 or data.get('architecture') != 'embedding':
            raise ValueError('Expected embedding schema v5')
        return cls(**data)

    def infer_shapes(self, batch_size=1, sequence_length=5):
        positive_int(batch_size, 'batch size')
        positive_int(sequence_length, 'sequence length')
        if self.positional != 'none' and sequence_length > self.max_length:
            raise ValueError('Sequence exceeds positional capacity')
        b, t, d, v, c = batch_size, sequence_length, self.embedding_dim, self.vocabulary_size, self.output_size
        rows = [dict(operator='Token IDs', shape=(b,t), weights=(), parameters=0),
                dict(operator='Embedding', shape=(b,t,d), weights=((v,d),), parameters=v*d)]
        if self.positional != 'none':
            learned = self.positional == 'learned'
            rows.append(dict(operator=self.positional+' position addition', shape=(b,t,d),
                             weights=((self.max_length,d),) if learned else (), parameters=self.max_length*d if learned else 0))
        rows.extend([dict(operator='Masked '+self.pooling+' pooling', shape=(b,d), weights=(), parameters=0),
                     dict(operator='Linear logits', shape=(b,c), weights=((c,d),(c,)), parameters=c*d+c)])
        return rows

    @property
    def parameter_count(self):
        return sum(row['parameters'] for row in self.infer_shapes(sequence_length=1))
