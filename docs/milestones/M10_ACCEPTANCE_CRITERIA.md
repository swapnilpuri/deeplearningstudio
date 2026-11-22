# M10 acceptance criteria and evidence

Implementation and automated/runtime checks complete, 2026-11-21 America/Chicago.
Independent educational acceptance and browser visual QA are separate pending checks.

| Requirement | Implementation / evidence | Result |
|---|---|---|
| M10.1 twelve navigation sections | learning/ui.py, transformer_content.py; all-section AppTest and live visits | Passed |
| M10.2 M8/synthetic/custom input integration | transformer_ui.py; M8 reuse and invalid JSON tests | Passed |
| M10.3 equal-head configuration, shapes, validation | TransformerSpec; configuration tests and UI warning flow | Passed |
| M10.4 independent head Q/K/V and selected weights | MultiHeadAttention traces and projection tables; shape/reference tests | Passed |
| M10.5 scaled attention and all masks | Calls M9 attention; strict PyTorch parity for 1/2/4 heads, causal and padding | Passed |
| M10.6 head heatmaps, side by side, distributions, masking and outputs | transformer_graphs.py; data/label/0–1 tests; live section checks | Passed (visual appearance unreviewed) |
| M10.7 concatenation, Wo, shapes | Exact concat and output tests; section tables | Passed |
| M10.8 controlled single/multi comparison | Identical packed weights, same input; actual M9 context shown before Wo; costs and no superiority claim | Passed |
| M10.9 residual | Shape guard, component toggle, equation tests | Passed |
| M10.10 LayerNorm | PyTorch, correction=0, epsilon/affine/constant-row tests and worked example | Passed |
| M10.11 FFN | ReLU/GELU, internal stages/parameters, outer-shape and position-independence tests | Passed |
| M10.12 correct pre/post blocks and stage inspector | Manual-equation parity, causal leakage and autograd tests; live post-norm stage selection | Passed |
| M10.13 adaptive diagram and residual paths | DOT for both orders, diagram data assertions, live registration | Passed (browser rendering unreviewed) |
| M10.14 deterministic explicit CPU training/reset | Adam/SGD tests, config invalidation/reset AppTest, default loss reduction; live Train/Reset | Passed |
| M10.15 actual parameter counts and complexity | 568-parameter reference case; inspected modules and cost tests | Passed |
| M10.16 twelve exercises, feedback, progress | Incomplete-answer validation, all correct, notes, completion, legacy import/export; live quiz | Passed |
| Backward compatibility | Complete M1–M10 regression: 330 passed | Passed |
| Type/compile/diff quality | ty, compileall, git diff --check | Passed |
| Requirements/design/guide/completion/handoff/roadmap | M10 milestone records and Volume 8 | Present |
| Feature branch, push and PR preparation | See handoff for final commit and PR | Delivery recorded separately |

No required functional feature is intentionally deferred. Optional custom head
widths and trained single-versus-multi accuracy comparisons are not implemented.
CPU is validated; optional GPU execution, fresh environment installation, browser
appearance, download dialogs and independent acceptance are not verified.
