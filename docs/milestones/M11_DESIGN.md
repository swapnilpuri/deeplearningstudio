# M11 architecture and implementation design

Baseline: fetched `deeplearningstudio/master` at `41e8cbf` (M10 PR #7 and
navigation PR #8 merged). Clean working tree; branch
`feature/m11-transformer-encoder-decoder`. Local master was stale and was not
rewritten. No pre-existing changes were present.

## Implementation batches

1. Inspect standards, merged baseline and shared navigation; create branch.
2. Reuse M8 positions and M10 self-attention/FFN in inspectable encoder layers.
3. Compose decoder layers, M9-based rectangular cross-attention and masks.
4. Add complete model, synthetic copy/reverse tasks, training and greedy decoding.
5. Add seven lazy shared tabs, tensor/architecture inspectors, quiz and progress.
6. Validate targeted/full regression and live runtime; document and prepare PR.

## Module boundaries and reuse

| Module | Responsibility |
| --- | --- |
| `learning/seq2seq_spec.py` | Frozen bounded model configuration; reuses M10 `TransformerSpec` and M8 validation |
| `learning/seq2seq_adapter.py` | M8 sinusoidal positions, learned positions, masks, encoder/decoder stacks and model traces |
| `learning/seq2seq_training.py` | Offline dataset, target shifting, PAD-aware loss, deterministic training and greedy generation |
| `learning/seq2seq_content.py` | Existing question-tuple convention, explanatory feedback and guided experiments |
| `visualization/seq2seq_graphs.py` | Rectangular Plotly attention matrices and configuration-driven Graphviz diagram |
| `learning/seq2seq_ui.py` | Lazy shared tabs, session-owned models/traces/results, controls and progress v1 |

M10 `MultiHeadAttention` performs encoder/decoder self-attention unchanged.
Cross-attention reuses its initialized packed Q/K/V/O parameters, with decoder
queries and encoder keys/values passed to M9 `attention`. M10 `FeedForward` and
`residual` are shared. No completed milestone's mathematics or navigation is
rewritten. Common residual/dropout/norm behavior resides in `ResidualLayer`;
encoder and decoder have distinct forward signatures.

## Mathematical conventions

Batch-first `[B,L,D]`; head tensors `[B,H,L,D/H]`; cross scores `[B,H,T,S]`.
Token embeddings and positions are added directly (no sqrt(D) embedding scaling).
Pre-norm: `x + dropout(sublayer(LN(x)))`. Post-norm:
`LN(x + dropout(sublayer(x)))`. There are two encoder and three decoder sublayers.
In pre-norm cross-attention only the decoder query stream is normalized; memory
is unchanged. There is no additional stack-final norm. Reference tests compare
equivalent PyTorch layers, not a different stack convention.

Dropout applies to combined inputs and residual branches; attention probability
dropout is deliberately absent. Reference parity uses dropout=0 and float64,
absolute/relative tolerance `1e-9`, mapping row-vector weights with transposes and
zeroing PyTorch attention biases. ReLU/GELU and both norm strategies are tested.

PAD=0, BOS=1, EOS=2; content IDs begin at 3. M8's UNK convention is not reused for
the separate synthetic vocabulary. Source streams include EOS, not BOS. Targets
start BOS, contain content, end EOS and then PAD. Padding masks mark forbidden
**keys** True. Visibility tensors `[B,Q,K]` mark permitted positions True and
are shared across heads. Encoder visibility is bidirectional; decoder combines
causality and target-key padding; cross-attention uses source-key padding only.
Padded query rows can attend valid keys; PAD labels are excluded from loss.
Fully masked rows raise clear errors before softmax. Masked logits contain
intentional negative infinity; model outputs/probabilities/gradients remain finite.

## State, training and performance

The existing `render_learning_section_navigation(SECTIONS, 'm11')` is used
directly; only `tab.open` panels render. All new widgets use `m11_` keys; inputs
use `persist_state='session'`. Progress stays in `learning_progress_v1` under
lesson `seq2seq`, preserving older lesson records and import/export format.

The demonstration model and last experiment trace are owned by the Streamlit
session and reused when configuration/input identity matches. Mutable models
are never globally cached. Training occurs only under `m11_train`; results stay
available across panels and control changes, with the saved configuration shown.
Reset removes training/generation/trace results. No model downloads or services.
On explicit progress import, the stored M11 note is queued and applied before
creating the notes editor on the next rerun. This replaces the old draft only
when importing, while preserving ordinary cross-tab drafts. The model diagram
uses vertical source/target paths and multiline nodes for readable labels.

Training uses 16 fixed synthetic examples (content length 1–3), seeded CPU
minibatch sampling, Adam/SGD, PAD-aware CE, norm clipping at 1, and full-set
evaluation after each update. CPU RNG is restored after model construction and
training; no process-global thread settings are changed. All reported metrics
are training-set metrics, not generalization evidence. Teacher-forced exact
match and autoregressive greedy exact match are separately identified.

Greedy decoding uses inference mode, restores the caller's train/eval setting,
encodes once, recomputes decoder prefixes, suppresses PAD/BOS for unfinished
examples, stops each sequence on EOS and pads completed examples while others
continue. A maximum generation length bounds unfinished predictions. It accepts
no future labels. Core model tensors can be moved with PyTorch `.to(device)`;
the studio and reproducibility checks are CPU-first, with no GPU UI or GPU claim.

## Scope and review boundaries

No schema migration, dependencies, frontend framework, pretrained models,
advanced positions, sampling, caching or later milestone implementation. M12's
old decoder-only roadmap wording is marked as absorbed into this expanded M11;
future M12 scope remains an owner decision. Desktop/narrow viewport, keyboard
and download-dialog acceptance must be reported separately from AppTest/runtime
checks. Exact evidence and remaining gaps belong in the completion report.
