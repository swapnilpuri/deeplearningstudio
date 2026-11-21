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
| Feature commit and PR against master | branch created; delivery pending |

Browser layout/hover/file-picker/download round-trip and independent acceptance
review are unverified. Automatic approval review blocked Chrome accessibility
inspection due to possible unrelated private tabs. No visual pass is claimed.
No merge authorized. Overall status: implementation/automated/runtime criteria
passed; full independent/manual acceptance remains pending.
