import io
import numpy as np
import pandas as pd
import pytest
from training.datasets import built_in_dataset, prepare_dataset, read_uploaded_dataset
from training.engine import Experiment, ExperimentConfig
from visualization.network_graph import network_figure

@pytest.mark.parametrize('name,classes', [('Two moons',2),('Iris',3),('Digits (8×8)',10)])
def test_builtin_and_split(name,classes):
    frame,target,task = built_in_dataset(name)
    bundle = prepare_dataset(frame, name=name, target=target, task=task)
    assert bundle.classification
    assert bundle.output_size == classes
    assert bundle.summary['total'] == sum(bundle.summary[s] for s in ('train','validation','test'))
    assert len(bundle.val_x)>0 and len(bundle.test_x)>0
    a=Experiment(ExperimentConfig(dataset=name,hidden=(8,),lr=.01),bundle)
    a.train_epochs(2)
    assert a.test_metrics()['samples']==len(bundle.test_x)
    assert 'val_loss' in a.history[-1]


def test_regression_and_csv():
    df = pd.DataFrame({'x':np.linspace(-1,1,100), 'target':np.linspace(-1,1,100)**2})
    parsed=read_uploaded_dataset(df.to_csv(index=False).encode(), 'sample.csv')
    bundle=prepare_dataset(parsed,name='uploaded',target='target',task='regression')
    exp=Experiment(ExperimentConfig(dataset='uploaded'),bundle)
    exp.train_epochs(3)
    assert exp.history[-1]['val_score']>=0


def test_excel_import():
    frame=pd.DataFrame({'x':[1,2,3],'y':[3,2,1]})
    buf=io.BytesIO()
    frame.to_excel(buf,index=False)
    imported=read_uploaded_dataset(buf.getvalue(),'sample.xlsx')
    pd.testing.assert_frame_equal(imported,frame)


def test_numeric_missing_and_training_only_scaler():
    rng=np.random.default_rng(4)
    values=rng.normal(size=300)
    frame=pd.DataFrame({'x':values, 'x2':values*2, 'target':['A']*150+['B']*150})
    frame.loc[2,'x']=np.nan
    bundle=prepare_dataset(frame,name='test',target='target',task='classification')
    assert np.isfinite(bundle.train_x.numpy()).all()
    assert np.allclose(bundle.train_x.numpy().mean(axis=0), 0, atol=.01)


def test_bad_features_and_label_distribution():
    frame=pd.DataFrame({'x':range(40),'y':['A']*39+['B']})
    with pytest.raises(ValueError):
        prepare_dataset(frame,name='bad',target='y',task='classification')
    with pytest.raises(ValueError):
        read_uploaded_dataset(b'data','bad.txt')


def test_large_layer_graph_compressed():
    fig=network_figure((784,16,10))
    assert len(fig.data)>0
    assert len(fig.data[1].x)==20
