# M5 implementation and verification report

Date: 2026-11-21. Starting commit: `7b1c0d1b779012e21384235852d42e2c23de4041`. Branch: `feature/m5-final-validation`. See [handoff](M5_HANDOFF.md) for revision/delivery status and [audit](M5_AUDIT.md) for baseline gaps.

## Implemented

- Twenty beginner topics with formulas, dimensions, numerical examples, logits/loss and gradients.
- Real PyTorch rectangular/RGB convolution: stride, padding, dilation, editable per-filter/per-channel kernels, persistent edits, bias, sampled-window highlights, channel contributions and pre/post activation maps.
- Max/average pooling, exact arithmetic, highlighted windows, formula assumptions and single-cell UI support.
- Configurable one-to-three-block builder, optional dense hidden layers, inferred Flatten width, weight/bias shapes, parameter counts, actual intermediate tensors/logits and JSON export.
- Reproducible Digits SGD/Adam mini-batch training, fixed /16 normalization, stratified disjoint splits, train/validation curves, convolution gradients, initial/trained filters and explicitly requested test evaluation with confusion matrix and mistakes.
- SGD/Adam one-step autograd inspection, input image, ordered code steps, before/gradient/after tensors and loss, deterministic numerical gradient check.
- Twelve quiz questions, six experiments with expected observations, preserved M4 progress JSON and dense lessons.
- Fixed CSV download data with shared exporter retaining validation, gradient and future metrics. Existing UTF-8 source fix retained.
- Removed tracked bytecode from Git without deleting local caches; added ignore rules.

## Executed verification

Windows host; project Python 3.12.12; torch 2.15.0+cpu; Streamlit 1.67.0; scikit-learn 1.9.1.

| Check | Result |
|---|---|
| Baseline full suite before source edits | 41 passed in 3.89s |
| Expanded suite | 70 passed in 13.14s on final rerun (earlier full run: 24.24s) |
| Edited-source type check | ty: all checks passed |
| CNN arithmetic | PyTorch comparison, rectangular/dilated RGB, shape/parameter counts, pooling, SGD/Adam and double-precision gradcheck |
| Leakage prevention | Disjoint indices, exact fixed /16 normalization, poisoned validation/test values leave learned weights identical for SGD and Adam |
| Headless AppTest | Four workspaces, both dense lessons, seven CNN sections, one-cell controls, training/evaluation/config invalidation |
| Running localhost Streamlit agent API | All workspaces/sections ready; RGB+dilation+bias, three blocks and two hidden dense layers, SGD/Adam, gradcheck, training and explicit final evaluation |
| Existing ONNX path | Actual dynamo=True export, onnx.load, checker and initializer check passed |

The first restricted expanded run passed 69 tests but ONNX fixture setup hit Windows temporary-directory permissions. The full run outside that restriction passed all 70; no tests were disabled or skipped.

## Acceptance status

**Implementation and automated/runtime acceptance passed. Browser/manual acceptance is unverified; M5 is not marked fully accepted.**

No known failing automated requirement remains. Browser layout, browser downloads/file-picker interactions, Netron, optional MNIST download, fresh uv installation and ONNX Runtime parity were not tested. The builder offers bounded teaching controls (one to three blocks, shared kernel/padding/pooling and optional dense widths); the neutral schema supports up to 32 operators, including rectangular/dilated convolution. Digits training settings are independent of builder experiments and the original Training Studio. CNN ONNX export and TensorFlow are outside M5.

A tiny one-epoch localhost smoke run produced 26.67% test accuracy. This verifies the evaluation flow, not model quality; avoid tuning against repeated test evaluations. Dependency ranges remain unlocked. Volumes 1 and 2 are preserved.

The actual Training Studio CSV HTTP download was also fetched and parsed successfully with validation and gradient columns. Browser visual checking was attempted but unavailable: browser host unavailable and Windows Computer Use could not confidently determine Chrome URL.
