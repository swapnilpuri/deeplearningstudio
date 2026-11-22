"""Bounded CPU experiment: learn contextual mean mixing across token positions."""
from dataclasses import dataclass, field
import math
import torch
from .transformer_spec import TransformerSpec
from .transformer_adapter import TransformerBlock
from .attention_adapter import visibility


@dataclass(frozen=True)
class TransformerTrainingConfig:
    spec: TransformerSpec = field(default_factory=TransformerSpec)
    steps: int = 80
    learning_rate: float = .01
    optimizer: str = 'Adam'
    causal: bool = False

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 300:
            raise ValueError('Use 1–300 training steps')
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('Learning rate must be finite in (0,1]')
        if self.optimizer not in ('Adam', 'SGD'):
            raise ValueError('Choose Adam or SGD')
        if not isinstance(self.causal, bool):
            raise ValueError('causal must be boolean')


def snapshot(trace: dict) -> dict:
    return {name: snapshot(value) if isinstance(value, dict) else value.detach().clone()
            for name, value in trace.items()}


def train_transformer(config: TransformerTrainingConfig) -> dict:
    model = TransformerBlock(config.spec)
    generator = torch.Generator().manual_seed(config.spec.seed + 100)
    x = torch.randn(8, 4, config.spec.d_model, generator=generator)
    mask = visibility(4, config.causal)
    routing = mask.float() / mask.sum(-1, keepdim=True)
    target = x + .5 * (routing @ x)
    # Post-norm necessarily normalizes final features; use a compatible target.
    if config.spec.norm == 'Post-LayerNorm':
        target = torch.nn.functional.layer_norm(target, (config.spec.d_model,))
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
    with torch.no_grad():
        before = snapshot(model(x, mask))
        losses = [float(torch.nn.functional.mse_loss(before['output'], target))]
    history, gradients = [], {}
    for step in range(config.steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x, mask)['output'], target)
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f'Missing or nonfinite gradient: {name}; reduce learning rate')
            gradients[name] = parameter.grad.detach().clone()
        optimizer.step()
        with torch.no_grad():
            after = model(x, mask)
            losses.append(float(torch.nn.functional.mse_loss(after['output'], target)))
        if not math.isfinite(losses[-1]):
            raise ValueError('Nonfinite training loss; reduce learning rate')
        history.append(dict(step=step+1, loss=losses[-1],
                            **{name: float(g.norm()) for name, g in gradients.items()}))
    return dict(config=config, X=x, target=target, before=before, after=snapshot(after),
                initial=initial, trained={name:p.detach().clone() for name,p in model.named_parameters()},
                gradients=gradients, history=history, losses=losses)
