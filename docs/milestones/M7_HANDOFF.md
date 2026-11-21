# M7 handoff

Status: implementation and automated/runtime validation passed; Git delivery in progress;
browser visual and independent acceptance pending. Branch: feature/m7-lstm-learning-studio.
Starting/current baseline: fcea18305f2112a2c34f2bcc8e48117f13d1d428, verified against
remote master on 2026-11-21. Working tree initially clean; M6 merge confirmed.
Requirements: M7_REQUIREMENTS.md; design: M7_DESIGN.md; decision: ../adr/ADR-0003-framework-neutral-lstm.md.

Baseline restricted test: 134 passed, 1 ONNX fixture permission error in 76.86s.
Normal Windows retry: 135 passed in 46.52s, no failures/skips.
Implemented: nine-section lesson with 24 concepts, real editable gate arithmetic,
unrolling and seven synchronized state/gate panels; controlled retention/writing/
exposure; neutral schema v4 with single/stacked/dropout/batch/time-first/final/sequence
heads; exact shape/count/JSON validation; memory-distance classification and isolated
sine forecasting; SGD/Adam with clipping; shared-data RNN comparison and measured
distance sweeps; full cell/hidden BPTT and optimizer updates; packed states/masks;
15-question quiz, notes, completion and legacy-compatible progress JSON.

Changed: learning/lstm_{spec,adapter,training,content,ui}.py; visualization/lstm_graphs.py;
learning/rnn_training.py (optional adapter/spec injection only); learning/ui.py (routing);
tests/test_lstm{,_ui}.py; Volume 5, ADR-0003, README, roadmap and all M7 records.
Existing AGENTS, Operating Model and prior volumes remain intact.

Focused recurrent computation: 80 passed in 11.84s (includes RNN regressions).
UI workflows: 16 passed in 69.23s. Final full suite: 181 passed in 26.55s,
no failures/errors/skips. Edited-source ty, compileall and diff whitespace passed.
Running localhost port 8767: 27 successful LSTM interactions over all nine sections,
plus initial app/workspace/lesson selection; no new console warnings or tracebacks.
One-epoch runtime memory CE 0.687311, accuracy 0.583333; two-horizon sine MSE
0.511409, MAE 0.597658. Workflow evidence only; no superiority/quality claim.

Browser attempt used the computer-use skill; automatic approval review rejected
Chrome accessibility inspection because the window may expose unrelated private
tabs. No browser input/inspection followed the rejection. User permission question
is pending. Rendered layout, chart hover, editor interaction and browser file-picker/
download round trip remain unverified. Fresh install, Netron and independent review
were not run. No known failing automated checks. Recurrent ONNX export and
variable-length task training are outside scope; packing/masking is demonstrated.

Next: finish feature commit/push and draft PR; then independent acceptance and
browser verification with suitable authorization. No automatic merge.
Resume: `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_lstm.py`;
`git status`; `git log -1`. Final commit will be resolved via `git rev-parse HEAD`.
