"""Digits CNN experiment with held-out test set and reproducible CPU training."""
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from .cnn_spec import digits_spec
from .cnn_adapter import CNNAdapter


@dataclass(frozen=True)
class CNNTrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = .01
    optimizer: str = 'Adam'
    seed: int = 42
    filters1: int = 8
    filters2: int = 16
    pooling: str = 'maxpool2d'

    def __post_init__(self):
        if any(not isinstance(v,int) or isinstance(v,bool) for v in (self.epochs,self.batch_size)) or not 1 <= self.epochs <= 100 or not 1 <= self.batch_size <= 512:
            raise ValueError('Invalid epochs/batch size')
        if not 0 < self.learning_rate <= 1 or self.optimizer not in ('SGD','Adam'):
            raise ValueError('Invalid optimizer/learning rate')
        if self.pooling not in ('maxpool2d','avgpool2d'): raise ValueError('Invalid pooling')
        digits_spec(self.filters1,self.filters2,self.pooling,self.seed)


def digits_splits(seed=42):
    """Split before fitting any preprocessing. Divide by fixed known pixel maximum 16.

    Stratified train/validation/test = 70%/15%/15% (approximately).
    Index arrays permit independent inspection of disjoint partitions.
    """
    ds=load_digits()
    idx=np.arange(len(ds.target))
    train_idx,hold_idx=train_test_split(idx,test_size=.30,random_state=seed,stratify=ds.target)
    valid_idx,test_idx=train_test_split(hold_idx,test_size=.5,random_state=seed,stratify=ds.target[hold_idx])
    def make(indices):
        x=torch.tensor(ds.images[indices].copy(),dtype=torch.float32).unsqueeze(1)/16.0
        y=torch.tensor(ds.target[indices].copy(),dtype=torch.long)
        return x,y
    return {'train':make(train_idx),'validation':make(valid_idx),'test':make(test_idx),
            'indices':{'train':train_idx,'validation':valid_idx,'test':test_idx}}


def train_digits(config: CNNTrainingConfig):
    split=digits_splits(config.seed)
    model=CNNAdapter().build(digits_spec(config.filters1,config.filters2,config.pooling,config.seed))
    initial={name:p.detach().clone() for name,p in model.named_parameters() if name.endswith('weight') and p.ndim==4}
    optim=(torch.optim.Adam(model.parameters(),lr=config.learning_rate) if config.optimizer=='Adam'
           else torch.optim.SGD(model.parameters(),lr=config.learning_rate))
    generator=torch.Generator().manual_seed(config.seed)
    loader=DataLoader(TensorDataset(*split['train']),batch_size=config.batch_size,shuffle=True,generator=generator)
    loss_fn=torch.nn.CrossEntropyLoss()
    history=[]
    for epoch in range(config.epochs):
        model.train(); running=0.;correct=0;seen=0;gnorm=0.;batches=0
        for x,y in loader:
            optim.zero_grad(set_to_none=True)
            logits=model(x);loss=loss_fn(logits,y);loss.backward()
            conv_grads=[p.grad.norm().item() for p in model.parameters() if p.grad is not None and p.ndim==4]
            gnorm+=sum(conv_grads);batches+=1
            optim.step()
            running+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();seen+=len(y)
        vx,vy=split['validation'];model.eval()
        with torch.no_grad():
            v_logits=model(vx)
            vloss=loss_fn(v_logits,vy).item()
            vacc=(v_logits.argmax(1)==vy).float().mean().item()
        history.append({'epoch':epoch+1,'train_loss':running/seen,'train_accuracy':correct/seen,
                        'val_loss':vloss,'val_accuracy':vacc,'conv_gradient_norm':gnorm/max(1,batches)})
    return {'model':model,'split':split,'history':history,'initial_filters':initial,
            'trained_filters':{k:p.detach().clone() for k,p in model.named_parameters() if k in initial}}


def evaluate_test(experiment):
    """Explicit, one-time final test evaluation; never used during training/tuning."""
    model=experiment['model'];x,y=experiment['split']['test'];model.eval()
    with torch.no_grad(): logits=model(x);prob=logits.softmax(1);pred=logits.argmax(1)
    return {'accuracy':(pred==y).float().mean().item(),
            'confusion_matrix':confusion_matrix(y.numpy(),pred.numpy(),labels=np.arange(10)),
            'predicted':pred.numpy(),'actual':y.numpy(),'confidence':prob.max(1).values.numpy(),
            'images':x.numpy()[:,0],'incorrect_indices':np.where(pred.numpy()!=y.numpy())[0]}
