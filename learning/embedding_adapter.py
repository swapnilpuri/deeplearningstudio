"""Token processing, real PyTorch embeddings and numerical experiments. No UI."""
from collections import Counter
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .embedding_spec import EmbeddingModelSpec, positive_int

PAD, UNK = '<PAD>', '<UNK>'
TOY_CORPUS = (
    'cat dog pet animal', 'dog cat pet home', 'cat kitten pet animal',
    'dog puppy pet animal', 'apple orange fruit food', 'orange apple fruit sweet',
    'apple pear fruit food', 'orange pear fruit sweet',
    'car bus road travel', 'bus train road travel', 'car train travel city',
) * 6


def tokenize(text, mode='word'):
    if not isinstance(text, str) or mode not in ('word', 'character'):
        raise ValueError('Text and word/character mode required')
    # Character mode retains spaces; word mode splits on whitespace (punctuation stays).
    return text.lower().split() if mode == 'word' else list(text.lower())


def build_vocabulary(corpus, mode='word'):
    counts = Counter(token for text in corpus for token in tokenize(text, mode))
    vocab = {PAD: 0, UNK: 1}
    for token in sorted(counts):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab, counts


def encode(tokens, vocabulary, length=None):
    if vocabulary.get(PAD) != 0 or vocabulary.get(UNK) != 1 or sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError('Vocabulary requires contiguous IDs with PAD=0 and UNK=1')
    ids = [vocabulary.get(token, 1) for token in tokens]
    if length is not None:
        positive_int(length, 'length')
        ids = ids[:length] + [0]*max(0, length-len(ids))
    return torch.tensor(ids, dtype=torch.long)


def validate_ids(ids, vocabulary_size):
    if not isinstance(ids, torch.Tensor) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Token IDs must be integer tensors')
    if ids.numel() and ((ids < 0).any() or (ids >= vocabulary_size).any()):
        raise ValueError('Token IDs outside vocabulary')


def embedding_table(vocabulary_size=12, embedding_dim=4, seed=42, initialization='normal', padding_idx=0, values=None):
    EmbeddingModelSpec(vocabulary_size, embedding_dim, padding_idx=padding_idx, seed=seed)
    if initialization not in ('normal', 'uniform', 'zeros'):
        raise ValueError('Unsupported initialization')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        table = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=padding_idx)
        with torch.no_grad():
            if initialization == 'uniform': table.weight.uniform_(-1, 1)
            if initialization == 'zeros': table.weight.zero_()
            if padding_idx is not None: table.weight[padding_idx].zero_()
            if values is not None:
                custom = torch.as_tensor(values, dtype=table.weight.dtype)
                if custom.shape != table.weight.shape or not torch.isfinite(custom).all():
                    raise ValueError('Custom weights must be finite with shape [V,D]')
                table.weight.copy_(custom)
    return table


def lookup_demo(table, ids):
    validate_ids(ids, table.num_embeddings)
    one_hot = F.one_hot(ids.long(), table.num_embeddings).to(table.weight.dtype)
    return dict(one_hot=one_hot, multiplication=one_hot @ table.weight, lookup=table(ids))


def pool_embeddings(vectors, mask=None, mode='mean'):
    if vectors.ndim != 3 or mode not in ('mean', 'sum', 'max'):
        raise ValueError('Expected [B,T,D] vectors and mean/sum/max pooling')
    mask = torch.ones(vectors.shape[:2], dtype=torch.bool, device=vectors.device) if mask is None else mask.bool()
    if mask.shape != vectors.shape[:2]: raise ValueError('Mask must have shape [B,T]')
    if vectors.shape[1] == 0: return vectors.sum(dim=1)
    valid = mask.unsqueeze(-1)
    if mode == 'max':
        result = vectors.masked_fill(~valid, -torch.inf).amax(dim=1)
        return torch.where(mask.any(dim=1, keepdim=True), result, torch.zeros_like(result))
    result = vectors.masked_fill(~valid, 0).sum(dim=1)
    return result / mask.sum(dim=1, keepdim=True).clamp_min(1) if mode == 'mean' else result


def sinusoidal_positions(length, dim):
    positive_int(length, 'length'); positive_int(dim, 'dimension')
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)*(-math.log(10000.)/dim))
    angles = positions*frequencies
    result = torch.zeros(length, dim)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :dim//2].cos()
    return result


class EmbeddingClassifier(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.embedding = nn.Embedding(spec.vocabulary_size, spec.embedding_dim, padding_idx=spec.padding_idx)
        self.positions = nn.Embedding(spec.max_length, spec.embedding_dim) if spec.positional == 'learned' else None
        self.head = nn.Linear(spec.embedding_dim, spec.output_size)

    def representations(self, ids):
        validate_ids(ids, self.spec.vocabulary_size)
        if ids.ndim != 2: raise ValueError('Token input must have shape [B,T]')
        self.spec.infer_shapes(ids.shape[0], max(1, ids.shape[1]))
        token = self.embedding(ids)
        position = torch.zeros_like(token)
        if self.positions is not None:
            position = self.positions(torch.arange(ids.shape[1], device=ids.device))[None].expand_as(token)
        elif self.spec.positional == 'sinusoidal' and ids.shape[1]:
            position = sinusoidal_positions(ids.shape[1], self.spec.embedding_dim).to(token)[None].expand_as(token)
        combined = token+position
        mask = torch.ones_like(ids, dtype=torch.bool) if self.spec.padding_idx is None else ids != self.spec.padding_idx
        pooled = pool_embeddings(combined, mask, self.spec.pooling)
        return dict(token=token, position=position, combined=combined, mask=mask, pooled=pooled, logits=self.head(pooled))

    def forward(self, ids):
        return self.representations(ids)['logits']


class EmbeddingAdapter:
    def build(self, spec):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            return EmbeddingClassifier(spec)


def gradient_experiment(optimizer='SGD', learning_rate=.1, seed=42):
    if optimizer not in ('SGD', 'Adam') or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Choose SGD/Adam and a positive finite learning rate')
    model = EmbeddingAdapter().build(EmbeddingModelSpec(6, 3, seed=seed)).double()
    ids = torch.tensor([[2, 2, 3, 0]])
    opt = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    before = model.embedding.weight.detach().clone()
    opt.zero_grad()
    trace = model.representations(ids)
    trace['token'].retain_grad()
    target = torch.tensor([1])
    loss = F.cross_entropy(trace['logits'], target)
    loss.backward()
    assert model.embedding.weight.grad is not None and trace['token'].grad is not None
    gradient = model.embedding.weight.grad.detach().clone()
    # Central difference on one used row, with all other parameters fixed.
    epsilon = 1e-5
    with torch.no_grad():
        model.embedding.weight[2, 0] += epsilon
        plus = F.cross_entropy(model(ids), target).item()
        model.embedding.weight[2, 0] -= 2*epsilon
        minus = F.cross_entropy(model(ids), target).item()
        model.embedding.weight.copy_(before)
    opt.step()
    with torch.no_grad(): after_loss = F.cross_entropy(model(ids), target).item()
    return dict(ids=ids, before=before, after=model.embedding.weight.detach().clone(), gradient=gradient,
                occurrence_gradients=trace['token'].grad.detach(), vectors=trace['token'].detach(),
                prediction=trace['logits'].detach(), target=target, loss=loss.item(), loss_after=after_loss,
                numerical_gradient=(plus-minus)/(2*epsilon))


def finite_array(values, ndim=None):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (ndim is not None and values.ndim != ndim):
        raise ValueError('Finite values with the expected rank required')
    return values


def normalize_vectors(values):
    values = finite_array(values)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)


def similarity(a, b):
    a, b = finite_array(a, 1), finite_array(b, 1)
    if a.shape != b.shape or not a.size: raise ValueError('Nonempty equal-sized vectors required')
    return dict(dot=float(a@b), cosine=float(normalize_vectors(a)@normalize_vectors(b)),
                euclidean=float(np.linalg.norm(a-b)), manhattan=float(np.abs(a-b).sum()))


def nearest_neighbors(query, candidates, top_k=5, metric='cosine', normalize=False, exclude=()):
    query, candidates = finite_array(query, 1), finite_array(candidates, 2)
    positive_int(top_k, 'top K')
    if candidates.shape[1] != query.size or metric not in ('cosine', 'dot'):
        raise ValueError('Compatible query/candidate dimensions and cosine/dot required')
    if normalize or metric == 'cosine':
        query, candidates = normalize_vectors(query), normalize_vectors(candidates)
    scores = candidates @ query
    ranked = [int(i) for i in np.argsort(-scores, kind='stable') if i not in exclude][:top_k]
    return [(i, float(scores[i])) for i in ranked]


def project_embeddings(values, dimensions=2, method='PCA', seed=42, perplexity=5.):
    values = finite_array(values, 2)
    if dimensions not in (2, 3) or min(values.shape) < dimensions:
        raise ValueError('Projection needs at least as many samples and features as output axes')
    if method == 'Direct':
        if values.shape[1] != dimensions: raise ValueError('Direct plots require matching embedding dimension')
        return values.copy()
    if method == 'PCA':
        from sklearn.decomposition import PCA
        return PCA(n_components=dimensions, svd_solver='full').fit_transform(values)
    if method == 't-SNE':
        if len(values) < 5 or not 0 < perplexity < len(values):
            raise ValueError('t-SNE requires >=5 samples and 0 < perplexity < sample count')
        from sklearn.manifold import TSNE
        return TSNE(n_components=dimensions, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto').fit_transform(values)
    raise ValueError('Unknown projection method')
