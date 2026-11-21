# M7 acceptance evidence

All mandatory criteria derive from M7_REQUIREMENTS.md. Automated/runtime evidence
is recorded below; passing unit checks do not establish browser visual acceptance.

| Criterion | Implementation / verification |
|---|---|
| Integrated lesson, all gate/state explorers | Passed: all 9 sections, all 24 concepts, cell navigation/JSON, gate interventions; 16 AppTest workflows + running localhost |
| Real PyTorch cell and gradients | Passed: independent LSTMCell output and input/weight gradient parity with/without bias at multiple dimensions |
| Builder, shapes, counts, JSON | Passed: 16 layer/layout/selection/bias combinations, nn.LSTM parity, stacked dropout eval trace and round trips |
| Memory classification and forecasting | Passed: SGD/Adam with stacked dropout, reproducibility and poisoned test/validation checks; both actual UI/runtime tasks |
| Controlled RNN/LSTM comparisons | Passed: identical data object/settings, measured validation distance sweep, explicit test evaluation; AppTest + localhost |
| BPTT, optimizer updates, clipping | Passed: retained cell/hidden gradients, per-gate norms, direct forget-path identity, exact SGD update and actual Adam execution |
| Variable lengths and masks | Passed: packed h/c vs individual runs, final-valid hidden equality, poisoned padding inputs/targets and zero padded gradients |
| M4–M6 and progress compatibility | Passed: 135-test baseline; full final 181 passed including legacy ONNX; progress round trip preserves prior lessons |
| Volume 5, roadmap, README, ADR, handoff | Implemented and reviewed for formulas, dimensions, worked examples and actual verification evidence |
| Feature commit and PR against master | Passed: pushed 8c23761535cae80d38bb2af7126110e914b272b2; draft PR #4 targets master; open and unmerged |

Browser verification on 2026-11-21 subsequently succeeded in a dedicated single-tab
Edge window at http://127.0.0.1:8767. Checked LSTM selection and section navigation,
cell Previous/Next, rendered gate arithmetic, reversed memory input/plots, two-layer
time-first sequence architecture (298 parameters), memory and forecast train/test,
RNN/LSTM training/comparison curves, SGD/Adam BPTT and clipping, invalid/valid packed
lengths, quiz/progress rendering, saved notes and actual JSON download. Downloaded
JSON was read back and contains schema v1 and the exact entered LSTM notes.

Still unverified in a browser: input-table cell editing, full quiz submission,
deliberate chart hover, specification download, and progress upload/load round trip.
The file picker opened, but automatic review rejected inspecting it because local
filenames could be exposed; it was dismissed without importing. Independent review
is pending. Prior browser host/URL restrictions were overcome for core workflows
by using the dedicated window; no full manual acceptance claim is made.
No merge authorized. Overall status: implementation/automated/runtime criteria
passed; full independent/manual acceptance remains pending.
