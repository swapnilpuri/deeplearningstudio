"""Framework-neutral LSTM schema v4; prior schemas remain unchanged."""
from dataclasses import asdict, dataclass
import math
from .rnn_spec import RNNModelSpec


@dataclass(frozen=True)
class LSTMModelSpec:
    input_size: int = 1
    hidden_size: int = 8
    num_layers: int = 1
    output_size: int = 2
    dropout: float = 0.
    selection: str = 'final'
    bias: bool = True
    batch_first: bool = True
    seed: int = 42
    schema_version: int = 4
    architecture: str = 'lstm'

    def __post_init__(self):
        RNNModelSpec(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,
                     output_size=self.output_size,selection=self.selection,bias=self.bias,
                     batch_first=self.batch_first,seed=self.seed)
        if isinstance(self.dropout,bool) or not isinstance(self.dropout,(int,float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be finite in [0,1)')
        if self.dropout and self.num_layers == 1:
            raise ValueError('Dropout applies between stacked layers; use at least two layers')
        if self.schema_version != 4 or self.architecture != 'lstm':
            raise ValueError('Expected LSTM schema v4')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != 4 or data.get('architecture') != 'lstm':
            raise ValueError('Expected LSTM schema v4')
        return cls(**data)

    def infer_shapes(self, batch_size=1, sequence_length=5):
        rows = RNNModelSpec(self.input_size, self.hidden_size, self.num_layers,
                            self.output_size, selection=self.selection, bias=self.bias,
                            batch_first=self.batch_first).infer_shapes(batch_size, sequence_length)
        h = self.hidden_size
        for layer in range(self.num_layers):
            width = self.input_size if layer == 0 else h
            rows[layer+1].update(operator=f'LSTM layer {layer+1}',
                parameters=4*h*width+4*h*h+(8*h if self.bias else 0),
                weights=((4*h,width),(4*h,h))+(((4*h,),(4*h,)) if self.bias else ()),
                hidden_shape=(1,batch_size,h), cell_shape=(1,batch_size,h))
        return rows

    @property
    def parameter_count(self):
        return sum(row['parameters'] for row in self.infer_shapes())
