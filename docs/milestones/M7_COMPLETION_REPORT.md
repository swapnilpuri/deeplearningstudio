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
| User-requested live recheck | 20 ready interactions in a fresh agent session on port 8767; all nine sections, both train/test tasks and RNN/LSTM train/test comparison; no app errors |

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

Browser inspection was attempted using the computer-use skill. Automatic review
initially rejected inspection because the window could expose unrelated private
tabs. The user explicitly approved local-app browser inspection. The resumed
attempt inspected the local-app window and entered the test-server address;
Computer Use then stopped because it could not confidently determine the browser
URL to enforce policy. No automated browser validation pass is claimed.
On the user's next verification request, the browser access host was unavailable
and automatic review rejected whole-window inspection of the multi-tab Chrome
window. A dedicated local-app-only window was requested; runtime checks continued
through the local agent API independently of browser access.

On 2026-11-21 the user reported starting Streamlit and supplied a screenshot.
It confirms the LSTM option and initial Concepts & mathematics screen displaying
RNN recap text/formula, the gate diagram, preview controls, and synchronized input,
forget, input-gate, candidate, cell, output-gate and hidden plots. This establishes
initial browser rendering only. Other section layouts, chart hover, editor actions,
browser upload/download round trip, fresh uv install, Netron and independent
architecture/acceptance review remain unverified. Prior M5/M6 manual acceptance
is also still pending. Headless success is not full visual acceptance.

Scope limits: unidirectional LSTM, no projection; per-step builder outputs but
fixed-length final-state task training; no autoregressive decoder or recurrent
ONNX export. No GRU internals/embeddings/attention/Transformers. Equal-width
comparison has unequal parameter counts; optional dropout is LSTM-only and
disclosed. Multi-seed performance conclusions were not tested or claimed.

**Acceptance: implementation, automated/runtime and core browser workflows passed;
remaining manual checks and independent acceptance pending.** See M7_ACCEPTANCE_CRITERIA.md and
M7_HANDOFF.md for criterion mapping, files and continuation commands.

## Git delivery

Implementation commit: `8c23761535cae80d38bb2af7126110e914b272b2`; pushed on
`feature/m7-lstm-learning-studio` and verified against the remote branch.
[Draft PR #4](https://github.com/swapnilpuri/deeplearningstudio/pull/4) targets master;
its open/draft/unmerged state and head/base commits were verified through GitHub.
The connector could read the PR but creation returned 403; CLI creation succeeded.
No merge performed. Final containing documentation commit: `git rev-parse HEAD`.
The follow-up changes documentation only; no source changes after the final full
regression, so tests were not repeated. Remaining acceptance checks are listed above.

## Dedicated browser verification — 2026-11-21

The user opened a dedicated single-tab Edge window at http://127.0.0.1:8767,
80% browser zoom. Native accessibility and screenshots then verified core browser
workflows, superseding the earlier browser-access limitation for these checks:

- Learning Studio/LSTM selection and navigation through the lesson sections.
- Cell Previous/Next changed the selected step 0 → 1 → 0; numerical gate matrices,
  activation/arithmetic tables rendered. Memory sequence reversal updated the plots
  and side-by-side cell/hidden heatmaps. No visible rendering error in inspected views.
- Two-layer I=2,H=4,C=2 builder: 298 parameters, time-first output (5,2,4), final
  hidden/cell (2,2,4), sequence linear head (5,2,2); diagram/tables rendered correctly.
- Actual browser memory training and explicit evaluation: 240 samples, length 12,
  distance 7, noise .05, seed 42, 5 epochs, batch 32, Adam .01, H=12, one layer,
  clip 1. Test CE 0.696761, accuracy 0.437500. Training plots and parameter changes shown.
- Switched to forecasting; stale results cleared. Same controls, horizon 1:
  test MSE 0.111243, MAE 0.292749. 733 parameters versus memory model's 746.
  These runs verify workflows, not general learning quality.
- RNN/LSTM comparison training produced actual overlaid train/validation curves
  and separate summaries: RNN 206 parameters, LSTM 746 at equal hidden width.
  No general superiority conclusion drawn from this five-epoch run.
- BPTT default SGD: loss 0.544184 → 0.479260; Adam: → 0.401718. Global gradient
  norm 6.953316 → 1.000000 with threshold 1. Gate-gradient table and state plots rendered.
- Packed lengths 0,2 displayed the validation warning; 6,2,4 restored execution
  with final hidden/cell shapes (1,3,3).
- Quiz/progress controls rendered; entered and saved experiment notes, downloaded
  learning-progress.json. Read back the downloaded JSON: schema_version=1 and
  exercise_notes.lstm exactly equals "Browser verification: hidden and cell states differ."

Computer-use scrolling/state capture timed out intermittently, then recovered by
reselecting the same dedicated window. No application defect or code change was
identified. The import file picker opened, but automatic review rejected its
inspection because unrelated local filenames might be exposed; it was dismissed.
Browser upload/load round trip remains unverified. Input-table cell editing, full
quiz submission, deliberate chart hover and spec download also remain unexecuted
in the browser (headless computation/workflow checks still passed as recorded).
Independent acceptance review remains pending. Test server left running for user use.
The final containing commit is resolved using `git rev-parse HEAD` because a
document cannot name its own containing commit. Delivery details will be appended
after the remote actions complete.
