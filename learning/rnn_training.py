"""Offline sequence datasets and deterministic training; held-out test is explicit."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from .rnn_spec import RNNModelSpec
from .rnn_adapter import RNNAdapter


def sequence_splits(task='classification', samples=240, sequence_length=8, noise=.05, seed=42, horizon=1):
    if task not in ('classification','forecasting','memory'):
        raise ValueError('Unknown sequence task')
    if type(samples) is not int or not 60 <= samples <= 3000 or type(sequence_length) is not int or not 2 <= sequence_length <= 60:
        raise ValueError('Use 60..3000 samples and length 2..60')
    if not math.isfinite(noise) or not 0 <= noise <= 1 or type(horizon) is not int or not 1 <= horizon <= 10:
        raise ValueError('Use noise 0..1 and horizon 1..10')
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('Invalid seed')
    rng = np.random.default_rng(seed)
    counts = (int(samples*.6),int(samples*.2),samples-int(samples*.6)-int(samples*.2))
    result = {'task':task,'indices':{},'source_series':{},'preprocessing':'Raw generated values; no fitted preprocessing.'}
    offset = 0
    for part,n in zip(('train','validation','test'),counts):
        if task == 'forecasting':
            # Each sample is a distinct independently phased source series. No
            # overlapping windows from any source can enter another split.
            times = np.arange(sequence_length+horizon)[None,:]
            phases = rng.uniform(-np.pi,np.pi,(n,1))
            frequencies = rng.uniform(.15,.45,(n,1))
            series = np.sin(times*frequencies+phases)+rng.normal(0,noise,(n,sequence_length+horizon))
            x, y = series[:,:sequence_length,None],series[:,sequence_length:]
        else:
            x = rng.normal(0,.35,(n,sequence_length,1))
            if task == 'classification':
                # Label is first < last; reversing the exact same elements flips
                # the label. An unordered bag cannot solve this task.
                y = (x[:,0,0] < x[:,-1,0]).astype(np.int64)
            else:
                y = rng.integers(0,2,n)
                x[:,0,0] = 2*y-1  # relevant first symbol, increasingly distant
        ids = np.arange(offset,offset+n); offset += n
        result['indices'][part] = ids
        result['source_series'][part] = ids.copy()
        result[part] = (torch.tensor(x,dtype=torch.float32),
                        torch.tensor(y,dtype=torch.float32 if task=='forecasting' else torch.long))
    return result


@dataclass(frozen=True)
class RNNTrainingConfig:
    task: str = 'classification'
    samples: int = 240
    sequence_length: int = 8
    noise: float = .05
    horizon: int = 1
    seed: int = 42
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = .01
    hidden_size: int = 12
    num_layers: int = 1
    optimizer: str = 'Adam'
    clip: float = 1.
    activation: str = 'tanh'

    def __post_init__(self):
        if self.task not in ('classification','forecasting','memory') or self.optimizer not in ('SGD','Adam'):
            raise ValueError('Invalid task/optimizer')
        for name,lo,hi in [('samples',60,3000),('sequence_length',2,60),('horizon',1,10),('epochs',1,100),('batch_size',1,512),('seed',0,2147483647)]:
            if type(getattr(self,name)) is not int or not lo <= getattr(self,name) <= hi:
                raise ValueError(f'Invalid {name}')
        if not math.isfinite(self.noise) or not 0 <= self.noise <= 1:
            raise ValueError('Invalid noise')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1 or not math.isfinite(self.clip) or self.clip < 0:
            raise ValueError('Invalid learning rate or clip (zero disables clipping)')
        RNNModelSpec(hidden_size=self.hidden_size,num_layers=self.num_layers,activation=self.activation)


def _metrics(model, x, y, task):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        if task == 'forecasting':
            loss = nn.functional.mse_loss(prediction,y).item()
            score = (prediction-y).abs().mean().item()
        else:
            loss = nn.functional.cross_entropy(prediction,y).item()
            score = (prediction.argmax(-1)==y).float().mean().item()
    return loss,score,prediction.detach()


def train_sequences(config: RNNTrainingConfig, data=None, *, spec=None, adapter=None):
    data = sequence_splits(config.task,config.samples,config.sequence_length,config.noise,config.seed,config.horizon) if data is None else data
    if data['task'] != config.task:
        raise ValueError('Dataset task differs from configuration')
    spec = spec if spec is not None else RNNModelSpec(hidden_size=config.hidden_size,num_layers=config.num_layers,
                        output_size=config.horizon if config.task=='forecasting' else 2,
                        activation=config.activation,seed=config.seed)
    model = (RNNAdapter() if adapter is None else adapter).build(spec)
    initial = {name:p.detach().clone() for name,p in model.named_parameters()}
    optimizer = (torch.optim.Adam if config.optimizer=='Adam' else torch.optim.SGD)(model.parameters(),lr=config.learning_rate)
    criterion = nn.MSELoss() if config.task=='forecasting' else nn.CrossEntropyLoss()
    x,y = data['train']; generator = torch.Generator().manual_seed(config.seed)
    history = []
    for epoch in range(config.epochs):
        model.train(); norms,clipped = [],[]
        order = torch.randperm(len(x),generator=generator)
        for ids in order.split(config.batch_size):
            optimizer.zero_grad()
            loss = criterion(model(x[ids]),y[ids]); loss.backward()
            norm = float(nn.utils.clip_grad_norm_(model.parameters(),config.clip if config.clip else float('inf')))
            if not math.isfinite(norm) or not torch.isfinite(loss):
                raise ValueError('Training diverged; reduce learning rate or enable clipping')
            norms.append(norm)
            clipped.append(float(torch.stack([p.grad.norm()**2 for p in model.parameters() if p.grad is not None]).sum().sqrt()))
            optimizer.step()
        train_loss,train_score,_ = _metrics(model,x,y,config.task)
        val_loss,val_score,_ = _metrics(model,*data['validation'],config.task)
        history.append({'epoch':epoch+1,'train_loss':train_loss,'validation_loss':val_loss,
                        'train_metric':train_score,'validation_metric':val_score,
                        'gradient_norm':float(np.mean(norms)),'clipped_norm':float(np.mean(clipped))})
    changes = {name:float((p.detach()-initial[name]).norm()) for name,p in model.named_parameters()}
    with torch.no_grad():
        hidden,_ = model.encode(x[:1])
    return {'config':config,'spec':spec,'model':model,'data':data,'history':history,
            'initial_parameters':initial,'parameter_changes':changes,'hidden':hidden[0]}


def evaluate_sequence_test(run):
    x,y = run['data']['test']; task = run['config'].task
    loss,score,prediction = _metrics(run['model'],x,y,task)
    if task == 'forecasting':
        errors = (prediction-y).square().mean(-1)
        mistakes = torch.argsort(errors,descending=True)[:10]
    else:
        mistakes = torch.where(prediction.argmax(-1)!=y)[0]
    return {'loss':loss,'metric':score,'metric_name':'MAE' if task=='forecasting' else 'accuracy',
            'prediction':prediction,'target':y,'mistakes':mistakes,'input':x}
