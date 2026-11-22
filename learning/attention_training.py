"""Explicit deterministic synthetic routing experiment, without Streamlit state."""
from dataclasses import dataclass
import math
import torch
from .attention_spec import AttentionSpec
from .attention_adapter import projections, self_attention


@dataclass(frozen=True)
class AttentionTrainingConfig:
    steps: int = 120
    learning_rate: float = .03
    optimizer: str = 'Adam'
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 500:
            raise ValueError('Use 1â€“500 training steps')
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('Learning rate must be finite in (0,1]')
        if self.optimizer not in ('SGD', 'Adam'):
            raise ValueError('Choose SGD or Adam')
        AttentionSpec(seed=self.seed)


def train_attention(config: AttentionTrainingConfig) -> dict:
    # Identity tokens make the target routing unambiguous, without language claims.
    x = torch.eye(4)
    target_attention = torch.eye(4).roll(1, dims=1)
    target_output = target_attention @ x[:, :2]
    initial = projections(AttentionSpec(4, 3, 2, config.seed))
    weights = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
    optimizer = getattr(torch.optim, config.optimizer)(weights.values(), lr=config.learning_rate)
    before = self_attention(x, weights)
    def objective(trace):
        return -(target_attention * trace['probabilities'].clamp_min(1e-12).log()).sum(-1).mean() + torch.nn.functional.mse_loss(trace['output'], target_output)
    history, gradients = [], {}
    losses = [float(objective(before).detach())]
    for step in range(config.steps):
        optimizer.zero_grad()
        trace = self_attention(x, weights)
        loss = objective(trace)
        loss.backward()
        gradients = {}
        for name, weight in weights.items():
            if weight.grad is None:
                raise ValueError(f'Missing gradient for {name}')
            gradients[name] = weight.grad.detach().clone()
        if not all(torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError('Nonfinite gradients; reduce learning rate')
        optimizer.step()
        with torch.no_grad():
            after = self_attention(x, weights)
            losses.append(float(objective(after)))
        history.append(dict(step=step + 1, loss=losses[-1], **{name: float(g.norm()) for name, g in gradients.items()}))
    return dict(config=config, X=x, target_attention=target_attention, target_output=target_output,
                initial=initial, trained={name: w.detach().clone() for name, w in weights.items()},
                before={name: t.detach().clone() for name, t in before.items()},
                after=after, gradients=gradients, history=history, losses=losses)
