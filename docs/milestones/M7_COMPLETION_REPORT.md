# M7 implementation and verification report

Date: 2026-11-21. Starting commit: `fcea18305f2112a2c34f2bcc8e48117f13d1d428`.
Initially clean master matches remote master (verified via git ls-remote), contains
the M6 merge. Branch: `feature/m7-lstm-learning-studio`. No user work overwritten.

## Implemented

- LSTM routing alongside all previous lessons; nine sections and 24 concepts with
  formulas, dimensions, numerical examples, use cases and live demonstrations.
- PyTorch cell explorer: editable inputs, packed matrices, both biases, initial
  h/c vectors, time navigation, selected-unit arithmetic, all four gate values,
  linear contributions, retained/written memory and synchronized plots.
- Clearly labelled fixed-gate experiments for forgetting, retention, writing,
  exposure, known impulse sequence, reversed order and direct cell gradients.
- Framework-neutral LSTM v4 specification; validation/shape/count/JSON support;
  single/stacked layers, inter-layer dropout, both layouts and final/per-step heads.
  Preserves dense v1/CNN v2/RNN v3/progress v1; actual construction stays in adapters.
- Offline memory classification with configurable signal distance/noise and
  isolated-source sine forecasting with multiple horizons; seeded SGD/Adam,
  clipping, train/validation metrics, examples/errors, counts, configuration and
  explicit held-out test evaluation. No fitted scaling or mandatory downloads.
- Reuses M6 training through optional spec/adapter injection, preserving defaults.
  RNN/LSTM comparison shares actual data/splits/budget and reports measured losses,
  gradients, parameters, durations, predictions and validation distance sweeps.
- Full gate-level BPTT: cell/hidden retained gradients, input gradients, packed
  gate weight/bias norms, actual SGD/Adam before/after parameters and recomputed loss.
  Clipping before/after norms and analytical fixed-gate path identity.
- Packed variable lengths with both final states, masks, final valid hidden states
  and padding-contamination checks. Fifteen quizzes, guided experiments, notes,
  completion and progress JSON preserve prior lesson records.
- Volume 5, README, roadmap, ADR-0003 and five M7 milestone documents.

## Executed evidence

Existing Windows Python 3.12 environment; torch 2.15.0+cpu, Streamlit 1.67.0.
No new dependencies, downloads, environment upgrades, ZIPs or duplicate repositories.

| Check | Actual result |
|---|---|
| Restricted baseline full pytest | 134 passed, 1 ONNX temp-fixture permission error in 76.86s |
| Normal Windows baseline retry | 135 passed in 46.52s |
| LSTM + existing RNN computation | 80 passed in 11.84s |
| LSTM AppTest workflows | 16 passed in 69.23s |
| Final complete regression | **181 passed in 26.55s**, no failures/errors/skips |
| Edited-source ty | Passed after correcting two nullable-gradient accesses |
| Compileall / whitespace / source structure | Passed |
| Actual localhost Streamlit | 27 LSTM ready interactions across all nine sections, plus app/workspace/lesson setup; no new server warnings or tracebacks |

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_lstm.py tests/test_rnn.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_lstm_ui.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
uv tool run ty check --python .venv learning/lstm_spec.py learning/lstm_adapter.py learning/lstm_training.py learning/lstm_ui.py learning/lstm_content.py learning/rnn_training.py learning/ui.py visualization/lstm_graphs.py
.\.venv\Scripts\python.exe -m compileall -q learning visualization tests
git diff --check
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8767 --server.enableAgentApi=true
```

The runtime used the installed agent API with its live OpenAPI description.
Initial cold load returned wait_timed_out; collecting the same session succeeded.
Interactions covered cell Previous/Next; memory reversal/forget changes; stacked
time-first sequence head/dropout; both actual train/test tasks; comparison train,
test and three-distance sweep; Adam/clipping; packed lengths 6,2,4; 15/15 quiz,
notes and completion. API flag was CLI-only, never saved in deployment settings.
One epoch, 60 examples: memory test CE **0.687311**, accuracy **0.583333**;
two-horizon forecast MSE **0.511409**, MAE **0.597658**. These are workflow smoke
measurements, not quality thresholds or claims that LSTM beats RNN.

## Verification boundaries

Passed: automated numerical/gradient/packing/serialization/count/split/training/
progress checks, all prior regressions and ONNX checker, headless UI and actual
running-app interactions. Failed: no known source/test failures. Skipped: none.
Initial environment issue: restricted Windows temp/cache access; normal-permission
execution resolved pytest and ty. No tests disabled or expectations weakened.

Browser inspection was attempted using the computer-use skill. Automatic approval
review rejected Chrome accessibility inspection because the window could expose
unrelated private tabs. No browser inspection/input followed that rejection; a
permission question was presented separately. Rendered layout/hover/editor,
browser upload/download round trip, fresh uv install, Netron and independent
architecture/acceptance review remain unverified. Headless success is not a
visual acceptance claim. Prior M5/M6 manual acceptance is also still pending.

Scope limits: unidirectional LSTM, no projection; per-step builder outputs but
fixed-length final-state task training; no autoregressive decoder or recurrent
ONNX export. No GRU internals/embeddings/attention/Transformers. Equal-width
comparison has unequal parameter counts; optional dropout is LSTM-only and
disclosed. Multi-seed performance conclusions were not tested or claimed.

**Acceptance: implementation and automated/runtime checks passed; full independent
and browser/manual acceptance pending.** See M7_ACCEPTANCE_CRITERIA.md and
M7_HANDOFF.md for criterion mapping, files and continuation commands.

## Git delivery

Feature commit, push and draft PR are being prepared against master. No merge.
The final containing commit is resolved using `git rev-parse HEAD` because a
document cannot name its own containing commit. Delivery details will be appended
after the remote actions complete.
