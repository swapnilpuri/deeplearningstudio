# M9 completion report

Date: 2026-11-21 (America/Chicago). Objective: bridge static M8 embeddings to
contextual mixtures through a usable single-head self-attention learning studio.
Implementation and automated/runtime validation complete; independent acceptance
and browser visual QA pending.

## Baseline and delivery

Started from clean local master d56d05d67f4dade5cb9472244b891b33e49c75d5,
which contains the M8 merge PR #5. Initial sandbox git fetch failed to connect;
normal-permission retry succeeded and verified remote master equals local master.
Created feature/m9-self-attention-learning-studio. No user changes overwritten,
no merge to master, no dependencies/downloads/services added.

## Features and architecture

Nine progressive lesson sections in existing Learning Studio navigation. Seeded
synthetic and M8 embedding lookup, actual session-trained M8 skip-gram table reuse,
custom embedding/projection JSON, token IDs, shape/dtype/value displays. Bias-free
row-vector projections; raw/scaled/masked logits, stable key-axis softmax, exact
causal/padding/combined visibility, row-sum and forbidden-probability reporting.
Query/key selectors and selectable score table explain coordinate dot products.
Plotly probability/raw/scaled/mask heatmaps and selected-query routing bars;
weighted value contributions and before/after representation tables; stage inspector.

Fixed cyclic routing experiment trains Wq/Wk/Wv through autograd with SGD/Adam.
Explicit Train/Reset, deterministic initialization, stale-config invalidation,
initial/trained weights, final pre-update gradients, gradient curves, parameter
deltas, loss and before/after routing. Nine checked exercises, notes, completion,
export/import using existing progress v1. Volume 7 explains motivations, shapes,
worked values, masking, scaling, gradients, interpretation and future multi-head use.

Core separation and mathematical decisions: [M9_DESIGN.md](M9_DESIGN.md).
No hidden bias or Linear transpose. True mask entries permit routing; all-blocked
rows raise a validation error. Padding blocks keys, not query rows. Intentionally
negative-infinite masked logits are displayed; valid probabilities/outputs remain
finite. Query/key width and value width may differ. Overflow is rejected rather
than silently producing invalid tensors. No untrained semantic-quality claims.

Changed/new source: learning/attention_spec.py, attention_adapter.py,
attention_training.py, attention_content.py, attention_ui.py; learning/ui.py;
visualization/attention_graphs.py; tests/test_attention.py and test_attention_ui.py.
Docs: README.md, docs/ROADMAP.md, docs/VOLUME_7_SELF_ATTENTION.md and the five M9
milestone records (requirements/design/acceptance/completion/handoff).

## Actual validation

Windows existing Python 3.12 environment; PyTorch 2.15.0+cpu, Streamlit 1.67.0.

| Check | Outcome |
|---|---|
| Initial core targeted suite | 14 passed in 3.08s |
| Initial core + UI suite | 26 passed in 11.12s |
| Expanded core + UI (M8 reuse/legacy progress/cell table) | 28 passed in 12.56s |
| Full M1–M9 regression, final source | **281 passed in 44.15s**, no failures/errors/skips |
| Initial edited-source ty | Four diagnostics (dict invariance and optional gradients), corrected |
| Final edited-source ty | All checks passed |
| compileall and git diff --check | Passed |
| Live localhost agent API | **23 ready snapshots**, no warning/exception elements |
| Server console after runtime checks | No additional warnings or tracebacks |
| Browser rendering and physical cell click | Not executed |
| Fresh installation/Netron/independent review | Not executed |

Initial restricted targeted runs exited successfully but logged Windows temp
cleanup PermissionError after pytest completion. Full suite used normal permissions
and had no cleanup trace. Initial ty invocation could not access existing uv cache;
normal-permission retry ran the checker, revealing and then verifying the fixes.
No tests were disabled or assertions weakened.

Commands actually executed (from repository root):

```powershell
git fetch deeplearningstudio
git rev-parse master deeplearningstudio/master
git switch -c feature/m9-self-attention-learning-studio
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_attention.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_attention.py tests/test_attention_ui.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
uv tool run ty check --python .venv learning/attention_spec.py learning/attention_adapter.py learning/attention_training.py learning/attention_content.py learning/attention_ui.py visualization/attention_graphs.py learning/ui.py
.\.venv\Scripts\python.exe -m compileall -q learning visualization tests
git diff --check
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8769 --server.enableAgentApi=true
```

Runtime checks read /_stcore/agent/openapi.json and POSTed JSON to
/_stcore/agent/interact using Python urllib: select Learning Studio and M9,
visit all nine sections, inspect score table, use causal mask, train default
120-step Adam routing, reset, answer all exercises, grade and mark complete.
Dataframe selection is action_unsupported in agent API: the actual click is
unverified, while query/key selector inspection and table registration are checked.

A separate executed default training measurement produced objective
**1.6328706741 → 0.0069277221**. Final pre-update gradient norms:
Wq 0.00933416, Wk 0.00667913, Wv 0.00174077. This describes one deterministic
synthetic task; it does not establish language quality or universal convergence.

## User instructions and limits

Run `.\.venv\Scripts\python.exe -m streamlit run app.py`, then choose
Learning Studio → Single-Head Self-Attention Learning Studio. Choose a section,
edit token input, widths, embeddings, projections and mask. Use query/key selectors
or the score table for cell arithmetic, and inspect routing/value contributions.
For learned input, train M8 Word embedding training first in the same session.
In M9 training, explicitly press Train; every press starts fresh. Reset restores
seeded initial weights. Exercises store shared progress; export JSON for persistence.

UI limits 1–24 tokens, 1–16 configured widths (M8/core <=32), <=500 steps.
Padding marks trailing positions as ignored keys; fully blocked rows are invalid.
No inherent position encoding, production tokenizer, language training, downloads,
ONNX attention export, architecture-builder schema, graph dependency or M10 scope.
Browser visual QA, direct cell clicking, fresh install and independent acceptance
remain unverified. No known source/test failures remain.

Acceptance matrix: [M9_ACCEPTANCE_CRITERIA.md](M9_ACCEPTANCE_CRITERIA.md).
Next: independent acceptance and M10 multi-head composition using the tested math
and trace structures, keeping value width explicit and mask semantics consistent.
Git outcome is recorded in [M9_HANDOFF.md](M9_HANDOFF.md).
