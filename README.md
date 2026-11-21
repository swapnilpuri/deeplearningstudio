# PyTorch Neural Network Studio v4 — Milestone 4

An interactive Streamlit learning application with four workspaces: **Neural Network Studio**, **Dataset Management**, **Training Studio**, and newly introduced **Learning Studio**. Built from the uploaded v3 source, not from a replacement scaffold.

## Requirements

- Windows 11 and Python **3.12**; `uv` package manager; CPU is sufficient for the small learning examples.
- Internet access for dependency installation, optional MNIST download, and Netron web viewing.

## Windows 11 install using uv

Extract the ZIP and locate the folder containing `app.py` and `requirements.txt`. Example PowerShell commands:

```powershell
cd 'C:\Personal\Swapnil\Learning\Innovation\neural-network-studio-v4'
uv venv --python 3.12
uv pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. If `uv` is missing, install it from [official documentation](https://docs.astral.sh/uv/getting-started/installation/). To enable optional MNIST downloads, install a compatible `torchvision` with `uv pip install torchvision`. For browser-based Netron inspection, open the exported ONNX in [Netron](https://netron.app).

**Notes:** PyTorch CPU-only builds may require using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your hardware. ZIP contains source code and requirements, **not** a Python virtual environment. The installer uses the published package index; exact versions are not locked in this milestone.

## Guided Learning Studio (new in v4)

1. Select **Learning Studio** from the sidebar.
2. Choose **Feed-Forward Neural Network** or **Deep Neural Network**.
3. Configure input size, hidden layer count/width, output width, activation, bias, seed and SGD learning rate.
4. Follow **Concepts & math** for formulas, **Architecture & tensors** for graph/weights, **Forward propagation** for real layer-by-layer PyTorch tensor outputs, and **Backward & SGD** for `loss.backward()` and one actual optimizer update.
5. Answer the quiz, run practice experiments, and write down observations. Mark the lesson complete and **Export progress JSON**; later restore with the import control.

**Progress:** held in Streamlit session state and optionally exported/imported as a JSON file. It is not a multi-user database, and progress is not synchronized across devices automatically. Quiz grading is multiple-choice; exercise notes are self-assessed.

**One-step distinction:** the Learning Studio independently rebuilds a seeded model each interaction and shows one SGD step. Persistent multi-epoch experimentation is in the existing Training Studio.

## Detailed documentation

- [Volume 1 — Feed-Forward Neural Networks](docs/VOLUME_1_FOUNDATIONS.md) — neurons, activations, matrix shapes, MSE, gradients, guided forward and backprop tutorials.
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

`ModelSpec` is a framework-neutral DTO. The adapter registry currently exposes **PyTorch only**. TensorFlow/Keras is planned for a future milestone, not shipped here. CNN, RNN, LSTM, embeddings, attention, Transformers and Mini GPT are also planned, not yet available.

## Tests and validation

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m compileall -q .
```

The automated suite includes v3 dataset, model, training and visualization tests, and v4 model specification, adapter, forward trace, autograd/SGD, progress and content tests. See `TEST_REPORT.md` for the author's actual verification results and coverage gaps. Streamlit UI interaction, Windows installation, ONNX runtime export, Netron inspection and MNIST download must be checked locally when dependencies/network are available.


## M5 — CNN Learning Studio (provisional overlay)

Select Learning Studio → Convolutional Neural Network (CNN). Explore PyTorch convolution,
pooling, CNN shape propagation, gradients and Digits training. See
`docs/VOLUME_3_CNN.md`, `docs/ROADMAP.md`, and
`docs/milestones/M5_COMPLETION_REPORT.md` for scope and verification status.
