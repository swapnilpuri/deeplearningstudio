# DeepLearningStudio — M9 Single-Head Self-Attention Learning Studio

An interactive Streamlit learning application with four workspaces: **Neural Network Studio**, **Dataset Management**, **Training Studio**, and **Learning Studio** with feed-forward, deep, convolutional, RNN, LSTM, embedding and single-head self-attention lessons.

## Requirements

- Windows 11 and Python **3.12**; `uv` package manager; CPU is sufficient for the small learning examples.
- Internet access for dependency installation, optional MNIST download, and Netron web viewing.

## Windows 11 install using uv

Use the existing repository checkout containing `app.py` and `requirements.txt`:

```powershell
cd 'C:\Personal\Swapnil\Learning\Innovation\neural-network-studio'
uv venv --python 3.12
uv pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. If `uv` is missing, install it from [official documentation](https://docs.astral.sh/uv/getting-started/installation/). To enable optional MNIST downloads, install a compatible `torchvision` with `uv pip install torchvision`. For browser-based Netron inspection, open the exported ONNX in [Netron](https://netron.app).

**Notes:** PyTorch CPU-only builds may require using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your hardware. The installer uses the published package index; exact versions are not locked in this milestone.

## Guided Learning Studio (new in v4)

1. Select **Learning Studio** from the sidebar.
2. Choose **Feed-Forward Neural Network** or **Deep Neural Network**.
3. Configure input size, hidden layer count/width, output width, activation, bias, seed and SGD learning rate.
4. Follow **Concepts & math** for formulas, **Architecture & tensors** for graph/weights, **Forward propagation** for real layer-by-layer PyTorch tensor outputs, and **Backward & SGD** for `loss.backward()` and one actual optimizer update.
5. Answer the quiz, run practice experiments, and write down observations. Mark the lesson complete and **Export progress JSON**; later restore with the import control.

**Progress:** held in Streamlit session state and optionally exported/imported as a JSON file. It is not a multi-user database, and progress is not synchronized across devices automatically. Quiz grading is multiple-choice; exercise notes are self-assessed.

**One-step distinction:** the Learning Studio independently rebuilds a seeded model each interaction and shows one SGD step. Persistent multi-epoch experimentation is in the existing Training Studio.

## Detailed documentation

- [Volume 6 — Embeddings and Representation Learning](docs/VOLUME_6_EMBEDDINGS.md) — tokenization, lookup, gradients, skip-gram, similarity/retrieval, sentence pooling, positions, classification, experiments and quizzes.
- [Volume 5 — LSTM](docs/VOLUME_5_LSTM.md) — gated memory, training, gradients and comparison tutorials.
- [Volume 1 — Feed-Forward Neural Networks](docs/VOLUME_1_FOUNDATIONS.md) — neurons, activations, matrix shapes, MSE, gradients, guided forward and backprop tutorials.
- [Volume 4 — Recurrent Neural Networks](docs/VOLUME_4_RNN.md) — recurrence, sequence tensors, BPTT, padding, classification and forecasting tutorials, experiments and quiz answers.
- [Volume 3 — Convolutional Neural Networks](docs/VOLUME_3_CNN.md) — convolution/pooling illustrations, tensor shapes, training workflow, backpropagation, tutorials and quiz explanations.
- [Volume 2 — Deep Neural Networks](docs/VOLUME_2_DEEP_NETWORKS.md) — deep layers, chain rule, gradient flow, activation/width comparisons, adapter design and experiments.

## Existing v3 workspaces (retained)

### Neural Network Studio

Configure architectures, inspect weights/biases and forward tensor values, generate Python model code and export ONNX models for inspection with Netron.

### Dataset Management

Choose synthetic data, Iris, Digits (8×8), optional MNIST, or upload CSV/XLSX. Preview columns; choose task and target; use numerical features, train-only preprocessing and a train/validation/test split. Apply a prepared dataset to Training Studio. Uploaded data is in server session memory; do not upload sensitive data to an untrusted host.

### Training Studio

Experiment with a PyTorch model, optimizers, batches, train/validation scores, gradient norms and learning curves. Evaluate the held-out test split, export trained model/code/metrics and ONNX. See the original v3 walkthrough preserved in [V3_GUIDE.md](docs/V3_GUIDE.md).

## Architecture and extensibility

```text
app.py (sidebar/workspace routing; existing v3 functionality)
  ├── models/feed_forward.py            # existing v3 architecture
  ├── training/engine.py                # existing v3 training
  ├── training/datasets.py              # existing v3 data workflows
  ├── visualization/network_graph.py    # existing v3 architecture diagrams
  ├── visualization/training_graphs.py  # existing v3 charts
  └── learning/ui.py                    # v4 Learning Studio frontend
        ├── learning/content.py         # lesson text, quizzes, exercises
        ├── learning/spec.py            # torch-independent immutable model spec
        ├── learning/pytorch_adapter.py # torch execution & autograd
        ├── learning/progress.py        # portable progress JSON
        └── visualization/learning_graphs.py # reusable tensor/gradient plots
```

`ModelSpec` is a framework-neutral DTO. The adapter registry currently exposes **PyTorch only**. TensorFlow/Keras is deferred. CNN uses schema v2, RNN v3, LSTM v4 and embedding v5, alongside unchanged dense v1 and progress v1. Attention, Transformers and Mini GPT remain planned.

## Tests and validation

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m compileall -q .
```

The automated suite includes v3 dataset, model, training and visualization tests, and v4 model specification, adapter, forward trace, autograd/SGD, progress and content tests. See `TEST_REPORT.md` for the author's actual verification results and coverage gaps. Streamlit UI interaction, Windows installation, ONNX runtime export, Netron inspection and MNIST download must be checked locally when dependencies/network are available.


## M5 — CNN Learning Studio

Choose Learning Studio → Convolutional Neural Network (CNN), then a CNN section:

- Concepts: twenty topics with worked formulas and dimensions.
- Convolution: rectangular grayscale/RGB inputs, editable per-filter/per-channel kernels, presets, stride/padding/dilation/bias, exact arithmetic and feature maps.
- Pooling: max/average windows, dimensions and highlighted contributions.
- Architecture: one to three convolution blocks, optional dense hidden layers, inferred Flatten, parameter shapes, actual forward tensors/logits and JSON export.
- Backpropagation: actual SGD/Adam updates, gradients, loss and optional numerical checking.
- Training: seeded Digits mini-batches, separate validation curves, initial/trained filters and explicitly requested held-out test evaluation.
- Quiz/progress: twelve questions, guided observations and M4-compatible JSON export/import.

The builder, one-step demonstration and training laboratory are independent experiments. Model definitions use framework-neutral CNN schema v2 beside unchanged dense schema v1. No TensorFlow/Keras adapter is included.

Baseline tests: **41 passed**. Expanded Windows suite: **70 passed**, including four-workspace headless UI checks and real dense ONNX export/checker. Running localhost Streamlit interactions and edited-source type checks also passed. Browser visual layout, downloads/file-picker interactions, Netron, optional MNIST download and fresh installation remain unverified; M5 awaits manual acceptance. See the current [completion report](docs/milestones/M5_COMPLETION_REPORT.md), [handoff](docs/milestones/M5_HANDOFF.md), [audit](docs/milestones/M5_AUDIT.md) and [roadmap](docs/ROADMAP.md). Historical `TEST_REPORT.md` records the earlier milestone, not current M5 verification.

## M6 — RNN Learning Studio

Select Learning Studio → Recurrent Neural Network (RNN). Nine sections cover 22 concepts; editable cell inputs/weights and step arithmetic; unrolled states; batch/time-first tensors; stacked recurrent architecture with final or sequence heads; offline order-sensitive, memory and sine datasets; SGD/Adam classification and forecasting; controlled BPTT/clipping/detach; variable lengths; and fourteen quiz questions with portable progress.

Training is explicit and independent of the architecture explorer. Changing settings clears stale models and test results. Held-out evaluation is a separate action; forecasting uses direct multi-horizon regression. Recurrent specs remain framework-neutral; dense/CNN formats and progress v1 remain compatible. M6 itself includes no recurrent ONNX export or gated-network lesson; M7 adds LSTM below.

## M7 — LSTM Learning Studio

Select Learning Studio → Long Short-Term Memory (LSTM). Nine sections cover 24 concepts,
editable packed-gate calculations and unrolling, controlled memory/state experiments,
stacked architectures and tensor inspection, offline memory classification and sine
forecasting, measured RNN/LSTM comparisons and dependency sweeps, full-state BPTT,
SGD/Adam updates and clipping, packed variable lengths, and 15 quizzes with progress.

Train/validation results are measured; held-out tests require an explicit action.
Equal-width comparisons show differing parameter costs and never guarantee LSTM
superiority. See [Volume 5](docs/VOLUME_5_LSTM.md), [M7 completion](docs/milestones/M7_COMPLETION_REPORT.md),
[handoff](docs/milestones/M7_HANDOFF.md) and [acceptance](docs/milestones/M7_ACCEPTANCE_CRITERIA.md)
for tutorials, verification evidence and remaining acceptance checks.

M5 baseline: **70 passed**. Final M6 regression: **135 passed** on Windows, including old ONNX checks and 65 new recurrent/runtime tests. See [M6 completion](docs/milestones/M6_COMPLETION_REPORT.md), [handoff](docs/milestones/M6_HANDOFF.md), [acceptance](docs/milestones/M6_ACCEPTANCE_CRITERIA.md) and [design](docs/milestones/M6_DESIGN.md) for final evidence and verification limits. Browser visual acceptance remains separate from headless and localhost runtime checks.

## M8 — Embedding Network Learning Studio

Select Learning Studio → Embedding Network. Twelve sections cover 22 concepts,
word/character tokenization and vocabularies, real nn.Embedding lookup and one-hot
agreement, gradients with finite differences and SGD/Adam, four similarity metrics,
exact retrieval, offline skip-gram negative sampling, initial/trained 2D/3D/PCA/t-SNE
plots, masked sentence pooling, learned/sinusoidal positions, static/contextual
explanations, a versioned architecture builder and supervised text classification.
Twenty-one quiz questions and guided experiments reuse progress schema v1.

Training is explicit, CPU-friendly and offline. Classifier vocabulary fits train
only; held-out evaluation is a separate action. Random vectors have no semantic
claims; decreasing toy loss is not evidence of broad semantic quality. Position
addition does not contextualize tokens. No new dependencies or attention/RAG.
See [Volume 6](docs/VOLUME_6_EMBEDDINGS.md), [M8 handoff](docs/milestones/M8_HANDOFF.md),
[completion evidence](docs/milestones/M8_COMPLETION_REPORT.md) and
[acceptance matrix](docs/milestones/M8_ACCEPTANCE_CRITERIA.md) for current status.

## M9 self-attention learning studio

Choose **Learning Studio → Single-Head Self-Attention Learning Studio**. Inspect
editable embeddings and Q/K/V, scores/scaling, causal/padding masks, normalized
routing and contextual outputs across nine progressive sections. Train the explicit
synthetic routing experiment with SGD/Adam; inspect gradients and reset weights.
Nine checked exercises reuse existing progress export/import. Everything runs
offline on CPU without model downloads. M8 trained skip-gram weights can be reused
from the same session. See [Volume 7](docs/VOLUME_7_SELF_ATTENTION.md),
[M9 acceptance](docs/milestones/M9_ACCEPTANCE_CRITERIA.md) and
[completion evidence](docs/milestones/M9_COMPLETION_REPORT.md).

## M11 Transformer encoder and decoder

Choose **Learning Studio → Transformer Encoder & Decoder** for seven shared
horizontal tabs covering fixed/learned positions, encoder/decoder stacks,
rectangular cross-attention, masks, teacher forcing, tiny offline copy/reverse
training, greedy generation, tensor/architecture inspection and fifteen quiz
checks. Training runs only on button press and results survive tab changes.
Defaults use a small CPU model and no dataset downloads. Accuracies describe
training examples, not held-out generalization. See [Volume 9](docs/VOLUME_9_ENCODER_DECODER.md),
[design](docs/milestones/M11_DESIGN.md),
[acceptance](docs/milestones/M11_ACCEPTANCE_CRITERIA.md), and
[execution evidence](docs/milestones/M11_COMPLETION_REPORT.md).

## M10 multi-head attention and Transformer blocks

Choose **Learning Studio → Multi-Head Attention & Transformer Blocks**. Twelve
sections expose independent Q/K/V head projections, shared-scale probability
heatmaps, causal/padding masks, concatenation/Wo, controlled single-head comparison,
residuals, affine LayerNorm, ReLU/GELU FFN and pre/post-norm block diagrams with
selectable tensor stages. Reuse M8 embeddings or provide a small custom matrix.

Explicit CPU training fits a cross-position synthetic mean task with SGD/Adam;
inspect loss, gradients, parameter changes and before/after head patterns. Reset
restores seeded initialization. Twelve exercises, notes and completion reuse
progress v1 JSON. Standard equal heads only; no stack, generation or downloads.
See [Volume 8](docs/VOLUME_8_TRANSFORMER_BLOCKS.md),
[acceptance](docs/milestones/M10_ACCEPTANCE_CRITERIA.md) and
[completion evidence](docs/milestones/M10_COMPLETION_REPORT.md).
