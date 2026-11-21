# DeepLearningStudio Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1–M3 | Neural Network Studio, Training Studio, Dataset Management | Regression and headless UI smoke passed; ONNX export/checker passed |
| M4 | Feed-forward and deep network lessons, progress JSON | Regression, legacy progress and both lesson runtime checks passed |
| M5 | CNN concepts, explorers, builder, PyTorch Digits training, gradient trace | Implemented; automated and running-app validation passed on Windows; browser/manual acceptance unverified |
| M6 | RNN cell/unrolling, recurrent schema, sequence datasets/training, BPTT, padding, quizzes, Volume 4 | Implemented; automated regression passed; browser visual acceptance unverified |
| M7 | LSTM gates/states, schema v4, builder, memory/forecast training, RNN comparison, BPTT, packing, Volume 5 | Merged; 181 automated tests and core browser workflows passed; remaining manual/independent acceptance pending |
| M8 | Embedding lookup/gradients, tokenization, skip-gram, retrieval, pooling, positions, schema v5, text classification, Volume 6 | Implemented; 253 regression tests and real-server workflows passed; M8 browser/manual and independent acceptance pending |
| M9 | Single-head self-attention | Planned |
| M10 | Multi-head attention | Planned |
| M11 | Transformer encoder | Planned |
| M12 | Transformer decoder | Planned |
| M13 | Mini GPT | Planned |
| Later | TensorFlow/Keras adapter | Deferred until PyTorch learning modules stable |

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
