# PyTorch Neural Network Studio v3

Complete standalone upgrade of v2: **Neural Network Studio**, **Dataset Management**, and **Training Studio**.

## Windows quick start with uv

Extract this ZIP into a separate directory, e.g. `C:\Personal\Swapnil\Learning\Innovation\neural-network-studio-v3`. Do **not** extract into a nested folder accidentally. Use PowerShell:

```powershell
cd C:\Personal\Swapnil\Learning\Innovation\neural-network-studio-v3
uv venv --python 3.12
uv pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you already have `.venv`, skip `uv venv`. Open http://localhost:8501.

**Optional MNIST**: install compatible torchvision with `uv pip install torchvision`. Selecting MNIST downloads the official training images via torchvision on first use (internet access required), saves them to `data/mnist`, and samples at most 3,000 examples for this teaching app. The other built-in datasets do not require downloads.

## Workspace 1 — Neural Network Studio

Build feed-forward architectures, inspect initialized weights/biases, trace the forward pass, export Python source or ONNX for Netron.

## Workspace 2 — Dataset Management

1. Select **Dataset Management** from the workspace radio in the sidebar.
2. Choose **Built-in / synthetic** (Two Moons, Concentric Circles, Sine, Linear, Iris, Digits, optionally MNIST), or **Upload CSV / Excel** (up to 12 MB / 20,000 rows).
3. Preview columns, class counts and a feature histogram.
4. Select the target column, task (classification/regression) and numeric feature columns (max 784). Class labels may be text; feature columns must contain numeric values or values convertible to numbers. Categorical feature encoding is not yet supported.
5. Choose missing-value imputation (median or mean), optional standard scaling, train/validation split percentages, and split seed. All normalization and imputation are fitted **only on the train split** to avoid leakage.
6. Click **Apply dataset to Training Studio**, then select **Training Studio** in the sidebar. The active dataset persists for this Streamlit browser session.

**Important**: Uploaded data is held in server session memory and is not automatically persisted to disk. Do not upload sensitive datasets to an untrusted deployment. Changing dataset, model hyperparameters or optimizer resets the experiment. Class-imbalanced tiny CSVs may fail stratified splitting; add examples to underrepresented classes.

## Workspace 3 — Training Studio: Two Moons tutorial

1. Dataset Management → **Two moons**, **320 samples**, **noise 0.12**, **seed 42**.
2. Leave target=`target`, task=`classification`, both features selected, standardize enabled, median imputation.
3. Set **70% train**, **15% validation**, **15% test**. Click **Apply dataset to Training Studio**. For 320 rows, this creates ~224 train / 48 validation / 48 test samples.
4. Switch to Training Studio. Choose **2 hidden layers**, **16 neurons each**, **Tanh**, **Adam**, **learning rate 0.02**, **batch size 32**, **weight decay 0**, **epochs per click 25**. Input=2, output=2 are derived from the active dataset.
5. Before training, inspect epoch 0 loss and train/validation accuracy. The random weights have not learned the moon pattern yet.
6. Click **Train 25 additional epochs**. Repeat until ~100 epochs; inspect **Data & predictions** (curved decision boundary), **Learning curves** (train and validation losses), **Gradients** (parameter gradient norms), **Model parameters** (learned weights).
7. After deciding your settings using validation, open **Export trained model** → **Evaluate test split** once for a held-out estimate. Test is not used during optimization. Avoid tuning based on repeated test results.
8. Download `.pt` checkpoint, training metrics `.csv`, config `.json` or **Generate ONNX** → **Download ONNX model** → open it at https://netron.app.

With 320 samples and a 70/15/15 split, actual split sizes depend on rounding and stratification. Accuracy is not guaranteed to monotonically increase; compare training and validation curves for overfitting. The test split is held out from training, but the app has not yet implemented automatic best-validation checkpoint selection or cross-validation.

## Dataset details

- **Synthetic**: Local NumPy generation with a deterministic seed. No download.
- **Iris**: Packaged scikit-learn classic dataset (150 samples; 4 features; 3 classes).
- **Digits (8×8)**: Packaged scikit-learn hand-written digits dataset (1,797 samples; 64 features; 10 classes).
- **MNIST (28×28)**: Optional torchvision download (784 pixel features; 10 classes). This version uses a fully connected network, not a CNN. Visualization shows a capped sample of neurons for large input layers.
- **CSV / Excel**: Numeric feature columns, categorical string class labels or a numeric regression target. Up to 12 MB upload; 20,000 rows; 784 features; 2–20 classes for classification.

## How training and evaluation work

- `training/datasets.py`: sources, file loading, numeric validation, reproducible train/validation/test splits, train-only imputation/standardization.
- `training/engine.py`: PyTorch model, SGD/Adam/RMSprop training, per-epoch **train and validation** metrics, gradient norms, manual held-out test evaluation.
- `models/feed_forward.py`: model builder/code generation.
- `visualization/*`: neuron diagram, training plots and decision boundary (only available for binary 2-feature classification).
- `app.py`: Streamlit workspaces and export controls.
- `tests/`: run `python -m pytest -v` (22 automated tests in the supplied build).

### Scope and limitations

This is an educational CPU-first app. There is no automatic best-validation checkpoint restore, categorical one-hot encoding for arbitrary input columns, time-series-specific splitting, class weighting, image CNNs, or production security/storage implementation yet. Avoid data leakage by ensuring your samples are independent; random splits are unsuitable for temporal, grouped or patient-level datasets without adapting the splitting strategy. MNIST downloads require internet; generated ONNX export requires the installed `onnx` and `onnxscript` packages. The browser UI and ONNX export have not been smoke-tested in the authoring environment.
