# M10 design

Baseline master `3baaea40a6dc184c5f580ccbd49eba0cec3a638a` includes merged M9 PR #6.
Remote fetch verified the same master. Feature branch:
`feature/m10-multihead-transformer-blocks`. Existing working tree was clean.

## Components

| Responsibility | Module |
|---|---|
| Frozen validated standard-head configuration | learning/transformer_spec.py |
| nn.Module MHA, FFN, block; trace/count/cost utilities | learning/transformer_adapter.py |
| Deterministic experiment independent of UI | learning/transformer_training.py |
| Twelve sections and checked questions | learning/transformer_content.py |
| Session controls and tensor inspectors | learning/transformer_ui.py |
| Plotly head adapters and adaptive DOT diagram | visualization/transformer_graphs.py |
| Existing navigation integration | learning/ui.py |

No change to M9 mathematical or UI public behavior, no new model/progress schema,
dependencies or generic framework. MHA calls M9 `attention` per batch/head.
Small educational bounds make this transparent approach adequate on CPU.
Efficient fused kernels are outside scope. Standard packed weights are [D,D],
row-vector convention, bias-free Q/K/V/O. Heads use contiguous projection columns;
traces have Q/K/V [B,H,T,D/H], scores/probabilities [B,H,T,T], heads [B,H,T,D/H],
concat/output [B,T,D]. Concatenation transposes head/time axes before reshaping.
All training/visualization traces come from these actual forward passes.

## Numerical decisions

M9 True visibility cells permit attention. [T,T] masks broadcast to every batch
and head; optional [B,T,T] masks allow batch-specific key padding. All-blocked
queries raise; padding excludes keys without deleting query outputs. Masked logits
intentionally contain negative infinity; probabilities, outputs and gradients must
be finite. Input and parameters share floating dtype/device. Model can be moved
with normal PyTorch `.to()`; UI and experiment are CPU-first.

`nn.LayerNorm(D, eps)` uses correction=0 variance across features. Trace includes
mean/variance/normalized output/gamma/beta; epsilon and affine controls in the
isolated demo do not silently alter the earlier block trace. FFN uses biased
Linear(D,F), activation, Linear(F,D), sharing weights over positions.

Pre: U=X+MHA(LN1(X)); Y=U+FFN(LN2(U)).
Post: U=LN1(X+MHA(X)); Y=LN2(U+FFN(U)).
No dropout, final pre-norm LayerNorm, positional encoding or stack is added.
Residuals reject unequal shapes instead of broadcasting. DOT solid edges are
the computational path; dashed edges bypass each sublayer. Selected stage values
are actual outputs, including raw post-norm residual sums.

Reference parity maps [Wq.T;Wk.T;Wv.T] into PyTorch in_proj_weight and Wo.T into
out_proj.weight, using bias=False and batch_first=True. Visibility is inverted for
PyTorch's disallow mask. Double precision checks use atol=rtol=1e-10.

## Experiment and lifecycle

Eight seeded random sequences, T=4. Target is X+0.5*A_uniform_visible@X, requiring
cross-position information. Causal mode uses prefix means. Post-norm target is
feature-normalized because that architecture normalizes its output. This is a
fixed-data fitting demonstration, not held-out evaluation or language learning.
All heads/FFN/affine norms receive gradients. Gradient snapshots precede the final
optimizer update; loss snapshots follow updates and include initial loss.

Only Train calls the training function. Reset discards results and reveals seeded
initial parameters; pressing Train always builds a fresh model. Frozen config
equality invalidates stale results when controls change. Initialization uses local
generators/fork_rng to preserve process RNG. State stores detached snapshots,
not models or optimizer graphs. Existing progress v1 stores lesson `transformer`.

## Counts and bounds

Actual trainable parameters are counted from modules. With biases only in FFN:
MHA=4D²; FFN=2DF+F+D; two affine LayerNorms=4D.
At D=8,F=16 the total is 256+280+32=568.
Per sequence QK and AV each use about T²D multiply-accumulates; projections 4TD²;
FFN 2TDF; score storage HT². Fixed D does not make more heads add parameters,
although score storage and softmax work grow. Bias/norm/activation costs omitted.

Limits: D<=32, H<=8, F<=128, B<=64, T<=24, steps<=300. Main path uses equal
key/query/value widths only. Ordinary random head patterns have no semantic claim.
