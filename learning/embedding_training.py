"""Deterministic offline skip-gram and supervised embedding experiments."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .embedding_adapter import TOY_CORPUS, build_vocabulary, tokenize, encode, EmbeddingAdapter
from .embedding_spec import EmbeddingModelSpec, positive_int


@dataclass(frozen=True)
class EmbeddingTrainingConfig:
    embedding_dim: int = 8
    window: int = 2
    negatives: int = 3
    learning_rate: float = .03
    batch_size: int = 32
    epochs: int = 5
    seed: int = 42
    optimizer: str = 'Adam'

    def __post_init__(self):
        for name in ('embedding_dim', 'window', 'negatives', 'batch_size', 'epochs'):
            positive_int(getattr(self, name), name)
        if self.embedding_dim > 64 or self.window > 5 or self.negatives > 20 or self.epochs > 100 or self.batch_size > 1024:
            raise ValueError('Configuration exceeds CPU laboratory limits')
        EmbeddingModelSpec(seed=self.seed)
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('Learning rate must be finite in (0,1]')
        if self.optimizer not in ('SGD', 'Adam'): raise ValueError('Choose SGD or Adam')


def context_pairs(corpus, vocabulary, window=2):
    positive_int(window, 'context window')
    pairs = []
    for text in corpus:
        ids = encode(tokenize(text), vocabulary).tolist()
        for i, target in enumerate(ids):
            for j in range(max(0, i-window), min(len(ids), i+window+1)):
                if i != j and target > 1 and ids[j] > 1: pairs.append((target, ids[j]))
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def sample_negatives(pairs, vocabulary_size, count, generator, positive_pairs=None):
    """Uniform eligible words; exclude every known positive context for each target."""
    positive_int(count, 'negative count')
    known = pairs if positive_pairs is None else positive_pairs
    positives = {}
    for target, context in known.tolist(): positives.setdefault(target, set()).add(context)
    result = []
    for target, context in pairs.tolist():
        forbidden = positives.get(target, set()) | {context, target, 0, 1}
        choices = torch.tensor([i for i in range(2, vocabulary_size) if i not in forbidden], dtype=torch.long)
        if not len(choices): raise ValueError('No eligible negative words; expand corpus or reduce context window')
        result.append(choices[torch.randint(len(choices), (count,), generator=generator)])
    return torch.stack(result) if result else torch.empty((0, count), dtype=torch.long)


class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, dim):
        super().__init__()
        self.input = nn.Embedding(vocabulary_size, dim, padding_idx=0)
        self.context = nn.Embedding(vocabulary_size, dim, padding_idx=0)

    def forward(self, targets, contexts, negatives):
        v = self.input(targets)
        positive = (v*self.context(contexts)).sum(-1)
        negative = (v[:, None, :]*self.context(negatives)).sum(-1)
        return -(F.logsigmoid(positive) + F.logsigmoid(-negative).sum(-1)).mean()


def train_skipgram(config, corpus=TOY_CORPUS):
    corpus = tuple(corpus)
    if sum(len(text) for text in corpus) > 20000: raise ValueError('Use a small corpus of at most 20,000 characters')
    vocabulary, counts = build_vocabulary(corpus)
    pairs = context_pairs(corpus, vocabulary, config.window)
    if not len(pairs): raise ValueError('Corpus must contain word context pairs')
    gen = torch.Generator().manual_seed(config.seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = SkipGram(len(vocabulary), config.embedding_dim)
    initial = model.input.weight.detach().clone()
    opt = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
    # A fixed evaluation negative set makes epoch-end objective comparisons meaningful.
    evaluation_negatives = sample_negatives(pairs, len(vocabulary), config.negatives, gen)
    example_negatives = evaluation_negatives[:8].clone()
    history = []
    for _ in range(config.epochs):
        order = torch.randperm(len(pairs), generator=gen)
        total = 0.
        for indices in order.split(config.batch_size):
            batch = pairs[indices]
            negatives = sample_negatives(batch, len(vocabulary), config.negatives, gen, pairs)
            opt.zero_grad()
            loss = model(batch[:,0], batch[:,1], negatives)
            loss.backward(); opt.step()
            total += loss.item()*len(indices)
        with torch.no_grad(): objective = model(pairs[:,0], pairs[:,1], evaluation_negatives).item()
        history.append(dict(train_loss=total/len(pairs), fixed_negative_loss=objective))
    return dict(config=config, model=model, vocabulary=vocabulary, counts=counts, pairs=pairs,
                negative_examples=example_negatives, initial=initial, trained=model.input.weight.detach().clone(),
                history=history, parameter_count=sum(p.numel() for p in model.parameters()))


def classification_splits(seed=42):
    """Split unique sentence records before fitting vocabulary; held-out records stay isolated."""
    positive = ('good', 'great', 'lovely', 'happy', 'bright', 'excellent')
    negative = ('bad', 'awful', 'poor', 'sad', 'dull', 'terrible')
    nouns = ('movie', 'book', 'meal', 'game', 'story', 'show', 'song', 'trip')
    records = [(f'{adjective} {noun}', label) for label, words in enumerate((negative, positive))
               for adjective in words for noun in nouns]
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(records), generator=gen).tolist()
    raw = {'train':[records[i] for i in order[:64]], 'validation':[records[i] for i in order[64:80]],
           'test':[records[i] for i in order[80:]]}
    # Explicit novel terms demonstrate UNK without teaching from validation/test text.
    raw['validation'].append(('unseenvalidation movie', 1))
    raw['test'].append(('unseentest book', 0))
    vocabulary, _ = build_vocabulary([text for text, _ in raw['train']])
    splits = {name:(torch.stack([encode(tokenize(text), vocabulary, 4) for text, _ in rows]),
                    torch.tensor([label for _, label in rows])) for name, rows in raw.items()}
    return dict(vocabulary=vocabulary, raw=raw, splits=splits)


def classification_metrics(model, ids, targets):
    model.eval()
    with torch.no_grad():
        logits = model(ids)
        predictions = logits.argmax(-1)
        confusion = torch.zeros(2, 2, dtype=torch.long)
        for target, prediction in zip(targets, predictions): confusion[target, prediction] += 1
        return dict(loss=F.cross_entropy(logits, targets).item(), accuracy=(predictions == targets).float().mean().item(),
                    confusion=confusion, predictions=predictions, logits=logits)


def train_classifier(config, data=None):
    data = classification_splits(config.seed) if data is None else data
    model = EmbeddingAdapter().build(EmbeddingModelSpec(len(data['vocabulary']), config.embedding_dim, seed=config.seed))
    initial = model.embedding.weight.detach().clone()
    opt = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
    gen = torch.Generator().manual_seed(config.seed)
    x, y = data['splits']['train']
    vx, vy = data['splits']['validation']
    history = []
    gradient = torch.zeros_like(initial)
    for _ in range(config.epochs):
        model.train()
        for indices in torch.randperm(len(y), generator=gen).split(config.batch_size):
            opt.zero_grad()
            loss = F.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            assert model.embedding.weight.grad is not None
            gradient = model.embedding.weight.grad.detach().clone()
            opt.step()
        train = classification_metrics(model, x, y)
        validation = classification_metrics(model, vx, vy)
        history.append(dict(train_loss=train['loss'], validation_loss=validation['loss'],
                            train_accuracy=train['accuracy'], validation_accuracy=validation['accuracy']))
    # No test predictions/metrics are computed until the explicit evaluation action.
    return dict(config=config, model=model, data=data, history=history, initial=initial,
                trained=model.embedding.weight.detach().clone(), gradient=gradient)


def evaluate_classifier_test(run):
    return classification_metrics(run['model'], *run['data']['splits']['test'])
