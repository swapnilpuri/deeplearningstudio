# DeepLearningStudio Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1–M3 | Neural Network Studio, Training Studio, Dataset Management | Regression and headless UI smoke passed; ONNX export/checker passed |
| M4 | Feed-forward and deep network lessons, progress JSON | Regression, legacy progress and both lesson runtime checks passed |
| M5 | CNN concepts, explorers, builder, PyTorch Digits training, gradient trace | Implemented; automated and running-app validation passed on Windows; browser/manual acceptance unverified |
| M6 | RNN cell/unrolling, recurrent schema, sequence datasets/training, BPTT, padding, quizzes, Volume 4 | Implemented; automated regression passed; browser visual acceptance unverified |
| M7 | LSTM gates/states, schema v4, builder, memory/forecast training, RNN comparison, BPTT, packing, Volume 5 | Merged; 181 automated tests and core browser workflows passed; remaining manual/independent acceptance pending |
| M8 | Embedding lookup/gradients, tokenization, skip-gram, retrieval, pooling, positions, schema v5, text classification, Volume 6 | Implemented; 253 regression tests, real-server and core browser workflows/file roundtrip passed; independent acceptance pending |
| M9 | Single-head self-attention, projections, masking, contextual outputs, routing training, exercises, Volume 7 | Implemented; validation evidence in M9 completion report; independent acceptance pending |
| M10 | Multi-head attention, residuals, LayerNorm, FFN, pre/post-norm block, deterministic training, Volume 8 | Implemented; 330 regression tests and 27 live-server snapshots passed; browser visual/independent acceptance pending |
| M11 | Transformer encoder and decoder, cross-attention, positions, tiny seq2seq training, Volume 9 | Implemented; validation and review status in M11 completion report |
| M12 | Scope to be defined after M11 review; previous decoder scope is included in expanded M11 | Awaiting architecture-owner scope decision |
| M13 | Mini GPT | Planned |
| Later | TensorFlow/Keras adapter | Deferred until PyTorch learning modules stable |

M11 expands the original encoder-only milestone according to its approved
implementation request, including the decoder foundation previously listed as
M12. It reuses M8–M10 mathematical components, standardized lazy horizontal tabs
and progress v1, with no later milestone implementation. See
[Volume 9](VOLUME_9_ENCODER_DECODER.md), [design](milestones/M11_DESIGN.md),
[acceptance map](milestones/M11_ACCEPTANCE_CRITERIA.md), and
[execution evidence](milestones/M11_COMPLETION_REPORT.md).

M5 uses a new framework-neutral CNN schema v2 alongside the unchanged dense schema v1. This intentionally avoids premature generic operator support for Transformer architecture.

M6 adds recurrent schema v3 without changing dense v1, CNN v2 or progress v1. Vanilla RNN computation is isolated from presentation. Sequence packing, state plots and deterministic dataset splits support future lessons without implementing gates or attention prematurely. M6 starts from the locally verified M5 feature tip; final Git delivery is recorded in its handoff. See [Volume 4](VOLUME_4_RNN.md).

M7 starts from verified merged master `fcea183`, adds a separate LSTM schema v4,
and reuses recurrent packing, training and plotting while preserving all earlier
formats. Explicit gate arithmetic matches PyTorch i,f,g,o ordering; fixed-gate
interventions are distinguished from learned gates. See [Volume 5](VOLUME_5_LSTM.md)
and [M7 evidence](milestones/M7_COMPLETION_REPORT.md).

M8 starts from verified merged master `dd878ca`. Dedicated embedding schema v5,
train-only vocabulary fitting, masked pooling and optional positional inputs
preserve earlier schemas and progress v1. Offline skip-gram and classification
provide measured learning examples, without pretrained downloads or attention.
See [Volume 6](VOLUME_6_EMBEDDINGS.md) and [M8 evidence](milestones/M8_COMPLETION_REPORT.md).

M9 reuses M8 embeddings and progress v1, adding framework-independent validation and
PyTorch differentiable single-head primitives. See [Volume 7](VOLUME_7_SELF_ATTENTION.md)
and [M9 evidence](milestones/M9_COMPLETION_REPORT.md). Multi-head composition remains M10.

M10 composes the existing M9 attention calculation into standard equally sized
heads and a single Transformer block. It reuses M8 inputs and progress v1, with
no changes to prior model schemas. See [Volume 8](VOLUME_8_TRANSFORMER_BLOCKS.md),
[design](milestones/M10_DESIGN.md) and [evidence](milestones/M10_COMPLETION_REPORT.md).
M11 can extend the validated block toward an encoder; generation and caching
remain later work.
