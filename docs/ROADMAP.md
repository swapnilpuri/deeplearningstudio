# DeepLearningStudio Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1–M3 | Neural Network Studio, Training Studio, Dataset Management | Regression and headless UI smoke passed; ONNX export/checker passed |
| M4 | Feed-forward and deep network lessons, progress JSON | Regression, legacy progress and both lesson runtime checks passed |
| M5 | CNN concepts, explorers, builder, PyTorch Digits training, gradient trace | Implemented; automated and running-app validation passed on Windows; browser/manual acceptance unverified |
| M6 | RNN cell/unrolling, recurrent schema, sequence datasets/training, BPTT, padding, quizzes, Volume 4 | Implemented; automated regression passed; browser visual acceptance unverified |
| Future | LSTM; embedding network | Planned |
| Future | Single-head attention; multi-head attention | Planned |
| Future | Transformer encoder; Transformer decoder; Mini GPT | Planned |
| Later | TensorFlow/Keras adapter | Deferred until PyTorch learning modules stable |

M5 uses a new framework-neutral CNN schema v2 alongside the unchanged v4 dense schema. This intentionally avoids premature generic operator support for Transformer architecture.

M6 adds recurrent schema v3 without changing dense v1, CNN v2 or progress v1. Vanilla RNN computation is isolated from presentation. Sequence packing, state plots and deterministic dataset splits support future lessons without implementing gates or attention prematurely. M6 starts from the locally verified M5 feature tip; final Git delivery is recorded in its handoff. See [Volume 4](VOLUME_4_RNN.md).
