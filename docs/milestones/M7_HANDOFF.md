# M7 handoff

Status: implementation, automated/runtime validation and Git delivery passed;
initial browser rendering confirmed by user screenshot; remaining browser interactions
and independent acceptance pending. Branch: feature/m7-lstm-learning-studio.
Starting baseline: fcea18305f2112a2c34f2bcc8e48117f13d1d428, verified against
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

Browser attempt used the computer-use skill. Initial automatic review rejected
inspection because the Chrome window might expose unrelated private tabs. The user
then explicitly approved local-app inspection. The resumed attempt selected the
local-app window and entered the test-server address, but Computer Use stopped
because it could not determine the browser URL confidently enough to enforce policy.
No automated browser acceptance pass is claimed. On 2026-11-21, the user started
Streamlit and supplied a screenshot confirming the LSTM learning-path entry and
Concepts & mathematics / RNN recap rendering, formula, gate diagram, preview sliders
and seven gate/state plots. This is static visual evidence for that screen only.
Other section layouts, chart hover, editor interaction and browser file-picker/
download round trip remain unverified. Fresh install, Netron and independent review
were not run. No known failing automated checks. Recurrent ONNX export and
variable-length task training are outside scope; packing/masking is demonstrated.

Implementation commit: `8c23761535cae80d38bb2af7126110e914b272b2`, pushed and verified
against the remote branch. [Draft PR #4](https://github.com/swapnilpuri/deeplearningstudio/pull/4)
targets master; open, unmerged. GitHub connector creation returned 403; GitHub CLI
created the PR successfully. Current containing documentation commit: resolve
`git rev-parse HEAD` (a document cannot name its own containing commit).

Next: independent acceptance review and remaining browser interaction checks.
No automatic merge. No source changes after the final 181-test regression;
this follow-up updates documentation only, so regression was not repeated.
Resume: `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_lstm.py`;
`git status`; `git log -1`. Final commit will be resolved via `git rev-parse HEAD`.
