"""Framework-neutral recurrent architecture, schema v3 (dense/CNN unchanged)."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RNNModelSpec:
    input_size: int = 1
    hidden_size: int = 8
    num_layers: int = 1
    output_size: int = 2
    activation: str = 'tanh'
    selection: str = 'final'
    bias: bool = True
    batch_first: bool = True
    seed: int = 42
    schema_version: int = 3
    architecture: str = 'rnn'

    def __post_init__(self):
        for name, limit in [('input_size',128), ('hidden_size',128), ('num_layers',4), ('output_size',128)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'{name} must be an integer in 1..{limit}')
        if self.activation not in ('tanh', 'relu') or self.selection not in ('final', 'sequence'):
            raise ValueError('Use tanh/relu and final/sequence selection')
        if type(self.bias) is not bool or type(self.batch_first) is not bool:
            raise ValueError('bias and batch_first must be boolean')
        if type(self.seed) is not int or not 0 <= self.seed <= 2147483647:
            raise ValueError('Invalid seed')
        if self.schema_version != 3 or self.architecture != 'rnn':
            raise ValueError('Unsupported recurrent schema')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != 3 or data.get('architecture') != 'rnn':
            raise ValueError('Expected recurrent schema v3')
        return cls(**data)

    def infer_shapes(self, batch_size=1, sequence_length=5):
        if any(type(v) is not int or v < 1 for v in (batch_size, sequence_length)):
            raise ValueError('Positive batch size and sequence length required')
        b, t, h = batch_size, sequence_length, self.hidden_size
        prefix = (b,t) if self.batch_first else (t,b)
        rows = [{'operator':'Sequence input', 'shape':(*prefix,self.input_size), 'parameters':0, 'weights':()}]
        width = self.input_size
        for layer in range(self.num_layers):
            weights = [(h,width), (h,h)] + ([(h,), (h,)] if self.bias else [])
            count = h*width + h*h + (2*h if self.bias else 0)
            rows.append({'operator':f'RNN layer {layer+1}', 'shape':(*prefix,h), 'parameters':count, 'weights':tuple(weights)})
            width = h
        selected = (b,h) if self.selection == 'final' else (*prefix,h)
        rows.append({'operator':self.selection+' state selection', 'shape':selected, 'parameters':0, 'weights':()})
        rows.append({'operator':'Linear head', 'shape':(*selected[:-1],self.output_size),
                     'parameters':h*self.output_size+(self.output_size if self.bias else 0),
                     'weights':((self.output_size,h),)+(((self.output_size,),) if self.bias else ())})
        return rows

    @property
    def parameter_count(self):
        return sum(row['parameters'] for row in self.infer_shapes())
