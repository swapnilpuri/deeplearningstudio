"""Reuse M6 training/evaluation and isolated source-series forecasting."""
from dataclasses import dataclass, replace
from time import perf_counter
import torch
from .rnn_training import RNNTrainingConfig, sequence_splits, train_sequences, evaluate_sequence_test
from .lstm_spec import LSTMModelSpec
from .lstm_adapter import LSTMAdapter


@dataclass(frozen=True)
class LSTMTrainingConfig(RNNTrainingConfig):
    task: str = 'memory'
    dependency_distance: int = 7
    dropout: float = 0.

    def __post_init__(self):
        super().__post_init__()
        LSTMModelSpec(hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout)
        if type(self.dependency_distance) is not int or not 1 <= self.dependency_distance < self.sequence_length:
            raise ValueError('Dependency distance must be 1..sequence_length-1')


def lstm_splits(config):
    data = sequence_splits(config.task,config.samples,config.sequence_length,config.noise,config.seed,config.horizon)
    if config.task == 'memory':
        # Terminal prediction must recall the symbol d steps earlier. Labels
        # and all distractors use independent RNG draws; only this symbol carries y.
        position = config.sequence_length-1-config.dependency_distance
        gen = torch.Generator().manual_seed(config.seed+1)
        for part in ('train','validation','test'):
            old,y = data[part]
            x = torch.randn(old.shape,generator=gen)*.35
            x[:,position,0] = 2*y.float()-1 + config.noise*torch.randn(len(y),generator=gen)
            data[part] = (x,y)
        data['signal_position'] = position
    return data


def train_lstm(config, data=None):
    data = lstm_splits(config) if data is None else data
    spec = LSTMModelSpec(hidden_size=config.hidden_size,num_layers=config.num_layers,
        output_size=config.horizon if config.task=='forecasting' else 2,seed=config.seed,dropout=config.dropout)
    start = perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        run = train_sequences(config,data,spec=spec,adapter=LSTMAdapter())
    run['seconds'] = perf_counter()-start
    with torch.no_grad():
        _,(hn,cn) = run['model'].encode(data['train'][0][:1])
    run['final_hidden'],run['final_cell'] = hn,cn
    return run


def compare_models(config):
    data = lstm_splits(config)
    start = perf_counter()
    rnn = train_sequences(config,data)
    rnn['seconds'] = perf_counter()-start
    lstm = train_lstm(config,data)
    return {'RNN':rnn,'LSTM':lstm}


def dependency_sweep(config, distances):
    """Measured validation evidence only; never read test data implicitly."""
    if config.task != 'memory':
        raise ValueError('Dependency sweeps require the memory task')
    rows = []
    for distance in distances:
        runs = compare_models(replace(config,dependency_distance=distance))
        for name,run in runs.items():
            rows.append(dict(model=name,distance=distance,
                validation_metric=run['history'][-1]['validation_metric'],
                parameters=run['spec'].parameter_count,seconds=run['seconds']))
    return rows
