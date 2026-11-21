# M5 audit and gap analysis

Date: 2026-11-21. Baseline: `7b1c0d1b779012e21384235852d42e2c23de4041` on master, clean working tree. M5 was already merged through PR #1. Development continues in the existing checkout on `feature/m5-final-validation`.

Before edits, the project Python 3.12 environment passed **41 tests in 3.89s**. System Python 3.14 could not collect tests because it lacks scikit-learn/Plotly; it is not the project interpreter.

Reviewed app.py, dense specs/adapter/UI/content/progress, CNN modules, training engine/datasets, visualizations, all existing tests and documentation.

| Requirement | Baseline gap | Resolution |
|---|---|---|
| Concepts/math | Nine brief grouped explanations | Twenty topics with formulas, dimensions, practical examples; both spatial formulas and variables |
| Convolution | Square only, no dilation control, bias always zero, first kernel edits only, no input highlight | Rectangular images/kernels, dilation arithmetic and sampled-position overlay, editable filter/channel kernels, persistent edits and bias control |
| Pooling | Correct numeric implementation; one-cell selection invalid | Shared shape function and guarded one-position UI controls |
| Architecture | Fixed two blocks and direct output head; shapes only | One to three blocks, hidden dense head, bias shapes, actual forward tensor/logit inspection |
| Training | Correct splits and fixed normalization; seed fixed in UI | Exposed seed, removed global seed mutation, held-out poisoning and normalization regression tests |
| Backpropagation | SGD only; no input display or interactive gradcheck | Input heatmap, ordered code, Adam comparison, deterministic input/weight/bias gradcheck |
| Framework compatibility | Separate neutral v2 CNN / v1 dense already correct | Preserve schemas; stricter integer dimension validation; compare shapes with actual tensors |
| Quiz/progress | Six questions, experiments without expected outcomes | Twelve questions, six guided observations, explicit M4 progress preservation test |
| CSV/UTF-8 | UTF-8 fix present; CSV populated one buffer but downloaded another empty buffer | Shared ordered-union exporter connected to download; real-history CSV regression |
| Testing/docs | 41 tests, stale overlay reports, missing handoff | Expanded math/runtime/ONNX regression coverage and current documentation |
| Git hygiene | Previously tracked bytecode, no ignore file | Untrack caches without deleting local files; ignore environments, secrets and generated assets |

Headless UI validation covers behavior, not rendered appearance. Browser appearance/download/file-picker interactions, Netron, optional MNIST, fresh installation and ONNX Runtime inference parity remain unverified. See completion and handoff reports.
