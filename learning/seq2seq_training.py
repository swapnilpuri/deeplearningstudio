"""Offline copy/reversal experiment, teacher forcing, and uncached greedy decoding."""
from dataclasses import dataclass, field
import math
import torch
from torch.nn import functional as F
from .embedding_adapter import validate_ids
from .embedding_spec import positive_int
from .seq2seq_spec import Seq2SeqSpec
from .seq2seq_adapter import EncoderDecoder

PAD, BOS, EOS = 0, 1, 2


def token_labels(ids):
    return [({PAD: 'PAD', BOS: 'BOS', EOS: 'EOS'}.get(int(i), f'T{int(i)-3}')) for i in ids]


def synthetic_data(spec: Seq2SeqSpec, count=16, task='copy', seed=None):
    positive_int(count, 'example count')
    if count > 64 or task not in ('copy', 'reverse'):
        raise ValueError('Use 1–64 examples and copy/reverse')
    generator = torch.Generator().manual_seed(spec.block.seed + 200 if seed is None else seed)
    length = min(3, spec.max_length-2)
    source = torch.full((count, length+1), PAD, dtype=torch.long)
    target = torch.full((count, length+2), PAD, dtype=torch.long)
    vocabulary = min(spec.source_vocabulary, spec.target_vocabulary)
    for i in range(count):
        n = int(torch.randint(1, length+1, (), generator=generator))
        content = torch.randint(3, vocabulary, (n,), generator=generator)
        source[i, :n], source[i, n] = content, EOS
        target[i, 0] = BOS
        target[i, 1:n+1] = content if task == 'copy' else content.flip(0)
        target[i, n+1] = EOS
    return dict(source=source, target=target)


def shift_targets(target):
    validate_ids(target, 64)
    if target.ndim != 2 or target.shape[1] < 2 or not (target[:, 0] == BOS).all():
        raise ValueError('Targets must be [B,T>=2] starting with BOS')
    for row in target:
        ends = (row == EOS).nonzero().flatten()
        if (len(ends) != 1 or ends[0] < 1 or (row[1:ends[0]] < 3).any()
                or (row[ends[0]+1:] != PAD).any()):
            raise ValueError('Targets require content, one EOS, then only PAD')
    return target[:, :-1], target[:, 1:]


def sequence_loss(logits, labels):
    if (logits.ndim != 3 or labels.shape != logits.shape[:2]
            or not torch.isfinite(logits).all() or not (labels != PAD).any()):
        raise ValueError('Finite [B,T,V] logits and aligned labels with at least one non-PAD token required')
    validate_ids(labels, logits.shape[-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=PAD)


def prediction_metrics(logits, labels):
    correct = logits.argmax(-1) == labels
    valid = labels != PAD
    return dict(token_accuracy=float(correct[valid].float().mean()),
                sequence_accuracy=float((correct | ~valid).all(-1).float().mean()))


def greedy_decode(model, source, max_new_tokens=None):
    limit = model.spec.max_length if max_new_tokens is None else max_new_tokens
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= model.spec.max_length:
        raise ValueError('Generation length must be within model max_length')
    training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            encoded = model.encode(source)  # Computed once, never reads target labels.
            target = torch.full((source.shape[0], 1), BOS, dtype=torch.long, device=source.device)
            done = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
            for _ in range(limit):
                logits = model.decode(target, encoded['memory'], source == PAD)['logits'][:, -1].clone()
                logits[:, PAD] = -torch.inf
                logits[:, BOS] = -torch.inf
                token = logits.argmax(-1).masked_fill(done, PAD)
                target = torch.cat([target, token[:, None]], dim=1)
                done |= token == EOS
                if done.all():
                    break
            return target
    finally:
        model.train(training)


def generation_accuracy(generated, target):
    width = max(generated.shape[1], target.shape[1])
    a = F.pad(generated, (0, width-generated.shape[1]), value=PAD)
    b = F.pad(target, (0, width-target.shape[1]), value=PAD)
    return float((a == b).all(-1).float().mean())


@dataclass(frozen=True)
class Seq2SeqTrainingConfig:
    spec: Seq2SeqSpec = field(default_factory=Seq2SeqSpec)
    steps: int = 150
    batch_size: int = 8
    learning_rate: float = .01
    optimizer: str = 'Adam'
    task: str = 'copy'

    def __post_init__(self):
        for value, name, maximum in ((self.steps, 'steps', 300), (self.batch_size, 'batch size', 32)):
            positive_int(value, name)
            if value > maximum:
                raise ValueError(f'{name} must be at most {maximum}')
        if (isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate)
                or not 0 < self.learning_rate <= .1):
            raise ValueError('Learning rate must be finite in (0,0.1]')
        if self.optimizer not in ('Adam', 'SGD') or self.task not in ('copy', 'reverse'):
            raise ValueError('Choose Adam/SGD and copy/reverse')


def train_seq2seq(config: Seq2SeqTrainingConfig):
    """Train on 16 fixed examples; all displayed accuracies are training-set metrics.

    fork_rng isolates seeded CPU dropout. Models are owned by the caller/session.
    No global thread settings or cached mutable model objects are modified.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.spec.block.seed)
        model = EncoderDecoder(config.spec)
        data = synthetic_data(config.spec, task=config.task)
        source, target = data['source'], data['target']
        decoder_input, labels = shift_targets(target)
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.spec.block.seed + 300)
        before = greedy_decode(model, source)
        history = []

        def evaluate(step, grad_norm):
            model.eval()
            with torch.no_grad():
                logits = model(source, decoder_input)
                loss = float(sequence_loss(logits, labels))
                metrics = prediction_metrics(logits, labels)
            history.append(dict(step=step, loss=loss, gradient_norm=grad_norm, **metrics))

        evaluate(0, 0.)
        for step in range(config.steps):
            model.train()
            indices = torch.randint(len(source), (config.batch_size,), generator=generator)
            optimizer.zero_grad()
            loss = sequence_loss(model(source[indices], decoder_input[indices]), labels[indices])
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError('Nonfinite gradients; reduce learning rate')
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            optimizer.step()
            evaluate(step+1, norm)
        after = greedy_decode(model, source)
        model.eval()
        return dict(config=config, model=model, data=data, before=before, after=after,
                    history=history, losses=[row['loss'] for row in history],
                    greedy_accuracy_before=generation_accuracy(before, target),
                    greedy_accuracy_after=generation_accuracy(after, target))
