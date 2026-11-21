# M6 acceptance criteria and evidence

Status: implementation and automated/runtime verification passed at the checkpoint; final Git delivery and browser visual acceptance are tracked separately in the completion report. No manual/visual pass is inferred from headless tests.

| Requirement | Implementation / evidence |
|---|---|
| Preserve M1–M5 workspaces, dense/CNN lessons | Only Learning Studio routing changed; full existing 70-test regression retained |
| 22 major concepts with explanation/formula/shapes/use/demo/experiment | rnn_content TOPICS, selectable concepts with live PyTorch preview, dedicated section experiments, Volume 4 |
| Cell recurrence and reference parity | cell_trace; both activations, nonzero initial state and biases compared with RNNCell and nn.RNN |
| Editable inputs/weights, selected step and navigation | Cell & unrolling; JSON matrices/biases, table inputs, initial-state control; AppTest navigation/dimensions/error feedback |
| Unrolled inputs/states, dependencies/shared weights, order effect | rnn_graphs; selected/current/previous highlights, hover values; original versus reversed final state; sequence-head outputs in architecture |
| Tensor layouts, full output versus h_n versus logits | Actual nn.RNN trace, batch-first/time-first, multiple layers, shape/count tests across 32 configurations |
| Neutral recurrent schema, final/sequence heads, serialization | RNNModelSpec v3, inference/counts without torch imports; separate adapter and ADR-0002; old specs round trip |
| Synthetic order-sensitive and sine datasets offline | First/last comparison, independent sine sources, optional first-symbol memory; reproducible IDs and tests |
| Leakage prevention | Disjoint source IDs; raw values/no fitted transform; poisoned validation/test do not alter learned parameters |
| Actual classification and regression training | SGD/Adam, single/stacked layers, epochs/batches/rate/clip; smoke tests and deterministic loss/quality checks |
| Train/validation/test metrics and predictions | Train/validation loss and accuracy/MAE; explicit final held-out action; incorrect labels/largest forecast errors |
| Hidden progression, gradients, parameter changes | Training charts and parameter norm changes; independent controlled BPTT view |
| BPTT shared gradients and SGD update | Per-step shared-weight contributions sum to autograd gradient; clipping identity and chain-product tests |
| Vanishing/exploding and truncation | Positive scalar ReLU gain controls; tanh saturation; detach boundaries with exact unreachable inputs tested |
| Variable lengths and padded-loss isolation | Packing, lengths, masks, final valid versus padded last state; independent per-sequence runs and poisoned padding; masked MSE tests |
| Reusable computation/presentation separation | rnn_adapter, rnn_training, rnn_graphs and rnn_ui separate; no recurrence math in Streamlit renderers |
| Quizzes and guided learning/progress | 14 questions, seven guided experiments; AppTest scores/notes/completion; v1 M4/M5 JSON preserved |
| Volume 4, README, roadmap and milestone records | VOLUME_4_RNN, M6 requirements/design/handoff/completion, acceptance table, ADR-0002 |
| Existing ONNX path | Full legacy export/checker test retained; recurrent export is outside this milestone |
| Running app | All nine RNN sections and 24 localhost agent API interactions passed, plus updated architecture rerun |
| Browser/Windows/installation checks | Tests run on actual Windows/Python 3.12; browser access host unavailable; browser visual/file-picker/download and fresh install unverified |
| Feature branch, commit, push and PR against master | See M6_HANDOFF and M6_COMPLETION_REPORT for final delivery; never merge automatically |

Final acceptance requires an independent review and any remaining manual workflow checks. No LSTM, GRU, attention or Transformer lesson is claimed. Existing M5 browser/manual acceptance was already unverified at the baseline.
