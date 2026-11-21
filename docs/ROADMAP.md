# DeepLearningStudio Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1–M3 | Neural Network Studio, Training Studio, Dataset Management | Existing in source repository; regression pending M5 integration |
| M4 | Feed-forward and deep network lessons, progress JSON | Existing master branch; regression pending M5 integration |
| M5 | CNN concepts, explorers, builder, PyTorch Digits training, gradient trace | Overlay implemented; UI/Windows/full-regression verification pending |
| Future | RNN; LSTM; embedding network | Planned |
| Future | Single-head attention; multi-head attention | Planned |
| Future | Transformer encoder; Transformer decoder; Mini GPT | Planned |
| Later | TensorFlow/Keras adapter | Deferred until PyTorch learning modules stable |

M5 uses a new framework-neutral CNN schema v2 alongside the unchanged v4 dense schema. This intentionally avoids premature generic operator support for Transformer architecture.
