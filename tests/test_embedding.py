"""Numerical/contract checks; smoke learning is not a semantic quality benchmark."""
import inspect
import json
import math
import numpy as np
import pytest
import torch
from learning.embedding_spec import EmbeddingModelSpec
from learning.embedding_adapter import (PAD, UNK, tokenize, build_vocabulary, encode, validate_ids,
    embedding_table, lookup_demo, pool_embeddings, sinusoidal_positions, EmbeddingAdapter,
    gradient_experiment, normalize_vectors, similarity, nearest_neighbors, project_embeddings)
from learning.embedding_training import (EmbeddingTrainingConfig, context_pairs, sample_negatives,
    train_skipgram, classification_splits, train_classifier, evaluate_classifier_test, SkipGram)


def test_tokenization_vocabulary_unknown_padding():
    assert tokenize('Cat  DOG!') == ['cat', 'dog!']
    assert tokenize('A b', 'character') == ['a', ' ', 'b']
    vocab, counts = build_vocabulary(['dog cat dog', 'cat pet'])
    assert vocab == {PAD:0, UNK:1, 'cat':2, 'dog':3, 'pet':4}
    assert counts == {'cat':2, 'dog':2, 'pet':1}
    assert encode(['cat','novel'], vocab, 4).tolist() == [2,1,0,0]
    assert encode(['cat','dog'], vocab, 1).tolist() == [2]
    assert encode([], vocab).dtype == torch.long
    with pytest.raises(ValueError): encode([], vocab, 0)
    with pytest.raises(ValueError): encode([], {'bad':3})
    with pytest.raises(ValueError): tokenize('x', 'BPE')


@pytest.mark.parametrize('ids',[torch.tensor([-1]), torch.tensor([4]), torch.tensor([1.]), torch.tensor([True])])
def test_invalid_ids(ids):
    with pytest.raises(ValueError): validate_ids(ids, 4)


@pytest.mark.parametrize('initialization',['normal','uniform','zeros'])
def test_real_lookup_one_hot_and_repeated(initialization):
    table = embedding_table(7, 3, initialization=initialization)
    ids = torch.tensor([[2, 2, 5, 0], [1, 3, 6, 0]])
    r = lookup_demo(table, ids)
    assert r['lookup'].shape == (2,4,3)
    torch.testing.assert_close(r['lookup'], r['multiplication'])
    torch.testing.assert_close(r['lookup'][0,0], r['lookup'][0,1])
    assert table.weight.numel() == 21 and table.weight[0].count_nonzero() == 0
    torch.testing.assert_close(table.weight, embedding_table(7,3,initialization=initialization).weight)
    custom = embedding_table(2,2, values=[[1,2],[3,4]])
    assert custom.weight.tolist() == [[1,2],[3,4]]  # custom padding row can be nonzero
    with pytest.raises(ValueError): embedding_table(values=[[float('nan')]])


@pytest.mark.parametrize('optimizer',['SGD','Adam'])
def test_gradients_accumulation_padding_unused_finite_difference_updates(optimizer):
    r = gradient_experiment(optimizer)
    assert math.isfinite(r['loss_after'])
    torch.testing.assert_close(r['gradient'][2], r['occurrence_gradients'][0,:2].sum(0))
    assert r['gradient'][[0,1,4,5]].count_nonzero() == 0
    torch.testing.assert_close(r['before'][[0,1,4,5]], r['after'][[0,1,4,5]])
    assert r['numerical_gradient'] == pytest.approx(r['gradient'][2,0].item(), abs=1e-9)
    if optimizer == 'SGD': torch.testing.assert_close(r['after'], r['before']-.1*r['gradient'])
    else:
        # First Adam step, no weight decay, bias-corrected moments reduce to g and g².
        torch.testing.assert_close(r['after'], r['before']-.1*r['gradient']/(r['gradient'].abs()+1e-8))


def test_similarity_normalization_and_rankings():
    r = similarity([1,2], [3,4])
    assert r['dot'] == 11 and r['cosine'] == pytest.approx(11/math.sqrt(125))
    assert r['euclidean'] == pytest.approx(math.sqrt(8)) and r['manhattan'] == 4
    assert similarity([0,0], [1,2])['cosine'] == 0
    assert np.array_equal(normalize_vectors([[0,0],[3,4]]), [[0,0],[.6,.8]])
    candidates = [[1,0],[10,10],[0,1]]
    assert nearest_neighbors([1,0], candidates, metric='dot')[0][0] == 1
    assert nearest_neighbors([1,0], candidates)[0][0] == 0
    assert nearest_neighbors([1,0], candidates, metric='dot', normalize=True) == nearest_neighbors([1,0], candidates)
    assert [i for i,_ in nearest_neighbors([1,0], candidates, 2, exclude=(0,))] == [1,2]
    assert [i for i,_ in nearest_neighbors([0,0], candidates, 3)] == [0,1,2]
    for a,b in (([1],[1,2]), ([float('nan')],[1])):
        with pytest.raises(ValueError): similarity(a,b)


@pytest.mark.parametrize('mode',['mean','sum','max'])
def test_masked_pooling_empty_padding_order_and_gradients(mode):
    vectors = torch.tensor([[[-2.,-3.],[-4.,-1.],[999.,999.]]], requires_grad=True)
    mask = torch.tensor([[True,True,False]])
    result = pool_embeddings(vectors, mask, mode)
    expected = {'mean':[[-3.,-2.]], 'sum':[[-6.,-4.]], 'max':[[-2.,-1.]]}[mode]
    torch.testing.assert_close(result, torch.tensor(expected))
    torch.testing.assert_close(result, pool_embeddings(vectors.flip(1),mask.flip(1),mode))
    result.sum().backward(); assert vectors.grad[0,2].count_nonzero() == 0
    assert pool_embeddings(vectors.detach(), torch.zeros_like(mask), mode).count_nonzero() == 0
    assert pool_embeddings(torch.empty(2,0,3), mode=mode).shape == (2,3)


@pytest.mark.parametrize('dim',[1,2,3,4,7])
def test_sinusoidal_even_odd_formula(dim):
    pe = sinusoidal_positions(6, dim)
    assert pe.shape == (6,dim)
    for position in range(6):
        for coordinate in range(dim):
            angle = position/(10000**(2*(coordinate//2)/dim))
            assert pe[position,coordinate] == pytest.approx(math.sin(angle) if coordinate%2==0 else math.cos(angle), abs=1e-6)


@pytest.mark.parametrize('positional',['none','learned','sinusoidal'])
@pytest.mark.parametrize('pooling',['mean','sum','max'])
@pytest.mark.parametrize('padding',[None,0,2])
def test_spec_shapes_counts_serialization_and_adapter(positional,pooling,padding):
    spec = EmbeddingModelSpec(9,3,4,padding,pooling,positional,7)
    assert EmbeddingModelSpec.from_dict(json.loads(json.dumps(spec.as_dict()))) == spec
    model = EmbeddingAdapter().build(spec)
    ids = torch.tensor([[2,3,0],[4,2,2]])
    r = model.representations(ids)
    assert tuple(model(ids).shape) == spec.infer_shapes(2,3)[-1]['shape']
    assert sum(p.numel() for p in model.parameters()) == spec.parameter_count
    torch.testing.assert_close(r['combined'],r['token']+r['position'])
    torch.testing.assert_close(model(ids),EmbeddingAdapter().build(spec)(ids))
    if positional == 'learned': torch.testing.assert_close(r['position'][0],model.positions(torch.arange(3)))
    if positional != 'none':
        with pytest.raises(ValueError): model(torch.zeros(1,8,dtype=torch.long))


def test_spec_validation_neutrality_and_old_formats():
    import learning.embedding_spec as module
    assert 'torch' not in inspect.getsource(module)
    for kwargs in ({'vocabulary_size':0}, {'embedding_dim':True}, {'padding_idx':-1}, {'padding_idx':12},
                   {'pooling':'bad'}, {'positional':'attention'}, {'schema_version':4}, {'seed':-1}):
        with pytest.raises(ValueError): EmbeddingModelSpec(**kwargs)
    with pytest.raises(ValueError): EmbeddingModelSpec.from_dict({'schema_version':4})
    from learning.rnn_spec import RNNModelSpec
    from learning.lstm_spec import LSTMModelSpec
    for cls in (RNNModelSpec,LSTMModelSpec): assert cls.from_dict(cls().as_dict()) == cls()


def test_context_generation_and_negative_correctness():
    vocab, _ = build_vocabulary(['a b c','d e f'])
    pairs = context_pairs(['a b c'],vocab,1)
    assert pairs.tolist() == [[2,3],[3,2],[3,4],[4,3]]
    gen = torch.Generator().manual_seed(2)
    negatives = sample_negatives(pairs,len(vocab),4,gen)
    for (target,context), row in zip(pairs.tolist(), negatives.tolist()):
        positives = {c for t,c in pairs.tolist() if t==target}
        assert not set(row) & (positives | {target,0,1,context})
    torch.testing.assert_close(negatives,sample_negatives(pairs,len(vocab),4,torch.Generator().manual_seed(2)))
    with pytest.raises(ValueError): sample_negatives(torch.tensor([[2,3],[3,2]]),4,1,gen)


@pytest.mark.parametrize('optimizer',['SGD','Adam'])
def test_skipgram_training_stability_reproducibility(optimizer):
    config = EmbeddingTrainingConfig(embedding_dim=3, epochs=2, batch_size=256, optimizer=optimizer)
    a,b = train_skipgram(config),train_skipgram(config)
    torch.testing.assert_close(a['trained'],b['trained'],rtol=0,atol=0)
    assert a['history'] == b['history']
    assert all(math.isfinite(row['train_loss']) for row in a['history'])
    assert not torch.equal(a['initial'],a['trained'])
    assert a['parameter_count'] == 2*len(a['vocabulary'])*3
    # Extreme logits still use stable logsigmoid rather than log(sigmoid()).
    model = SkipGram(5,2)
    with torch.no_grad(): model.input.weight.fill_(1000); model.context.weight.fill_(-1000)
    loss = model(torch.tensor([2]),torch.tensor([3]),torch.tensor([[4]]))
    assert torch.isfinite(loss); loss.backward(); assert torch.isfinite(model.input.weight.grad).all()


@pytest.mark.parametrize('optimizer',['SGD','Adam'])
def test_classifier_split_leakage_training_and_explicit_test(optimizer):
    data = classification_splits()
    assert 'unseenvalidation' not in data['vocabulary'] and 'unseentest' not in data['vocabulary']
    raw = data['raw']
    sets = [set(text for text,_ in raw[name]) for name in ('train','validation','test')]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    vocab, _ = build_vocabulary([text for text,_ in raw['train']]); assert vocab == data['vocabulary']
    assert data['splits']['test'][0][-1,0] == 1
    config = EmbeddingTrainingConfig(epochs=2, optimizer=optimizer)
    a,b = train_classifier(config,data),train_classifier(config,data)
    torch.testing.assert_close(a['trained'],b['trained'],rtol=0,atol=0)
    assert a['history'] == b['history'] and 'test' not in a
    # Poison held-out labels; training must be exactly unaffected.
    poisoned = {**data, 'splits':{**data['splits'], 'test':(data['splits']['test'][0],torch.full_like(data['splits']['test'][1],99))}}
    torch.testing.assert_close(a['trained'],train_classifier(config,poisoned)['trained'],rtol=0,atol=0)
    result = evaluate_classifier_test(a)
    assert math.isfinite(result['loss']) and 0 <= result['accuracy'] <= 1
    assert result['confusion'].sum() == len(raw['test'])
    assert a['gradient'][0].count_nonzero() == 0


def test_projection_validation_and_seed():
    values = np.random.default_rng(42).normal(size=(10,4))
    a = project_embeddings(values)
    assert a.shape == (10,2) and np.array_equal(a,project_embeddings(values))
    assert project_embeddings(values,3).shape == (10,3)
    assert np.array_equal(project_embeddings(values[:,:2],method='Direct'),values[:,:2])
    with pytest.raises(ValueError): project_embeddings(values[:3],method='t-SNE')
    with pytest.raises(ValueError): project_embeddings(values,method='t-SNE',perplexity=10)


def test_progress_preserves_m4_through_m7():
    from learning.progress import load_progress,export_progress,mark_complete,record_quiz
    old = dict(schema_version=1,completed=['feed_forward','deep','cnn','rnn','lstm'],quiz_scores={'lstm':{'correct':3,'total':4}},exercise_notes={'rnn':'keep'})
    new = record_quiz(mark_complete(load_progress(json.dumps(old)),'embedding'),'embedding',3,4)
    loaded = load_progress(export_progress(new))
    assert set(loaded['completed']) == set(old['completed'])|{'embedding'}
    assert loaded['quiz_scores']['lstm'] == old['quiz_scores']['lstm'] and loaded['exercise_notes'] == old['exercise_notes']
