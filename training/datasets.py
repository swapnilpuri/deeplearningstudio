"""Dataset registry, safe tabular imports, train-only preprocessing and deterministic splits."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_ROWS = 20000
MAX_FEATURES = 784

@dataclass
class DatasetBundle:
    name: str
    task: str
    features: list[str]
    classes: list[str]
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    raw_frame: pd.DataFrame
    target: str
    summary: dict

    @property
    def classification(self) -> bool:
        return self.task == 'classification'

    @property
    def input_size(self):
        return self.train_x.shape[1]

    @property
    def output_size(self):
        return len(self.classes) if self.classification else 1


def read_uploaded_dataset(content: bytes, filename: str) -> pd.DataFrame:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('Upload a nonempty file of at most 12 MB.')
    suffix = filename.lower().rsplit('.', 1)[-1]
    if suffix == 'csv':
        frame = pd.read_csv(BytesIO(content), nrows=MAX_ROWS + 1)
    elif suffix == 'xlsx':
        frame = pd.read_excel(BytesIO(content), engine='openpyxl', nrows=MAX_ROWS + 1)
    else:
        raise ValueError('Only CSV and XLSX files are supported.')
    if len(frame) > MAX_ROWS:
        raise ValueError(f'Datasets may contain at most {MAX_ROWS:,} rows.')
    if frame.shape[1] < 2:
        raise ValueError('Dataset needs at least one feature column and a target column.')
    if frame.columns.duplicated().any():
        raise ValueError('Dataset has duplicate column names.')
    return frame


def built_in_dataset(name: str, *, seed: int = 42, samples: int = 320, noise: float = .12) -> tuple[pd.DataFrame, str, str]:
    """Return frame, target column, and task; synthetic data is regenerated on demand."""
    if name == 'Iris':
        data = load_iris(as_frame=True)
        frame = data.frame.copy()
        frame['target'] = [data.target_names[i] for i in data.target]
        return frame, 'target', 'classification'
    if name == 'Digits (8×8)':
        data = load_digits(as_frame=True)
        frame = data.frame.copy()
        frame['target'] = [str(i) for i in data.target]
        return frame, 'target', 'classification'
    if name in ('Linear regression', 'Sine regression', 'Two moons', 'Concentric circles'):
        from training.engine import ExperimentConfig, make_dataset, is_classification
        cfg = ExperimentConfig(dataset=name, seed=seed, samples=samples, noise=noise)
        x, y = make_dataset(cfg)
        frame = pd.DataFrame(x.numpy(), columns=[f'x{i+1}' for i in range(x.shape[1])])
        frame['target'] = y.numpy().reshape(-1)
        return frame, 'target', 'classification' if is_classification(name) else 'regression'
    if name == 'MNIST (28×28)':
        try:
            from torchvision.datasets import MNIST
        except ImportError as exc:
            raise ValueError('Install torchvision to load MNIST: uv pip install torchvision') from exc
        try:
            ds = MNIST(root='data/mnist', train=True, download=True)
        except Exception as exc:
            raise ValueError(f'MNIST download failed: {exc}') from exc
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(ds), size=min(samples, len(ds)), replace=False)
        images = ds.data[chosen].numpy().reshape(len(chosen), -1).astype('float32') / 255.0
        frame = pd.DataFrame(images, columns=[f'pixel_{i}' for i in range(784)])
        frame['target'] = [str(int(n)) for n in ds.targets[chosen]]
        return frame, 'target', 'classification'
    raise ValueError(f'Unknown built-in dataset {name}')


def prepare_dataset(frame: pd.DataFrame, *, name: str, target: str, task: str,
                    seed: int = 42, train_fraction: float = .7,
                    val_fraction: float = .15, normalize: bool = True,
                    impute: str = 'median', features: Optional[list[str]] = None) -> DatasetBundle:
    """Split first, impute and scale fitted on training data ONLY to avoid leakage."""
    if task not in ('classification', 'regression'):
        raise ValueError('Task must be classification or regression')
    if not (.5 <= train_fraction <= .9 and .05 <= val_fraction <= .3 and train_fraction + val_fraction <= .95):
        raise ValueError('Use valid train/validation/test fractions with at least 5% test data.')
    if target not in frame.columns:
        raise ValueError('Target column not found')
    if features is None:
        features = [str(c) for c in frame.columns if c != target]
    if not features or len(features) > MAX_FEATURES or target in features or len(set(features)) != len(features):
        raise ValueError('Select 1–784 unique feature columns (not including target).')
    if not set(features).issubset(frame.columns):
        raise ValueError('Some selected feature columns are missing.')
    if len(frame) < 30 or len(frame) > MAX_ROWS:
        raise ValueError('Dataset must have 30–20,000 rows.')
    working = frame.loc[:, [*features, target]].copy()
    working = working.dropna(subset=[target])
    if len(working) < 30:
        raise ValueError('Not enough rows with a target value.')
    x = working[features].apply(pd.to_numeric, errors='coerce')
    if x.isna().all(axis=0).any():
        raise ValueError('Each selected feature must have at least one numeric value.')
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.isna().all(axis=0).any():
        raise ValueError('Each feature must have at least one finite numeric value.')
    if impute not in ('mean', 'median'):
        raise ValueError('Imputation must be mean or median')
    if task == 'classification':
        labels = LabelEncoder()
        y = labels.fit_transform(working[target].astype(str))
        classes = list(map(str, labels.classes_))
        if not (2 <= len(classes) <= 20):
            raise ValueError('Classification needs 2–20 distinct target classes.')
        counts = np.bincount(y)
        if counts.min() < 4:
            raise ValueError('Each class needs at least four examples for splitting.')
    else:
        y = pd.to_numeric(working[target], errors='coerce').to_numpy(dtype='float64')
        if not np.isfinite(y).all():
            raise ValueError('Regression target must be numeric and finite.')
        classes = []
    indices = np.arange(len(working))
    stratify = y if task == 'classification' else None
    try:
        train_idx, remain_idx = train_test_split(indices, train_size=train_fraction, random_state=seed, stratify=stratify)
        frac_val_remaining = val_fraction / (1.0 - train_fraction)
        val_idx, test_idx = train_test_split(remain_idx, train_size=frac_val_remaining,
                                              random_state=seed+1, stratify=y[remain_idx] if task == 'classification' else None)
    except ValueError as exc:
        raise ValueError(f'Unable to produce stratified splits: {exc}. Try more samples or a larger validation fraction.') from exc
    imputer = SimpleImputer(strategy=impute, keep_empty_features=True)
    tr = imputer.fit_transform(x.iloc[train_idx])
    va = imputer.transform(x.iloc[val_idx])
    te = imputer.transform(x.iloc[test_idx])
    if normalize:
        scaler = StandardScaler()
        tr = scaler.fit_transform(tr)
        va = scaler.transform(va)
        te = scaler.transform(te)
    if not (np.isfinite(tr).all() and np.isfinite(va).all() and np.isfinite(te).all()):
        raise ValueError('Preprocessed features contain invalid values.')
    def tensor_x(v):
        return torch.as_tensor(np.array(v, dtype=np.float32))
    def tensor_y(idx):
        if task == 'classification':
            return torch.as_tensor(y[idx], dtype=torch.long)
        return torch.as_tensor(y[idx].reshape(-1, 1), dtype=torch.float32)
    return DatasetBundle(
        name=name, task=task, features=features, classes=classes,
        train_x=tensor_x(tr), train_y=tensor_y(train_idx),
        val_x=tensor_x(va), val_y=tensor_y(val_idx),
        test_x=tensor_x(te), test_y=tensor_y(test_idx),
        raw_frame=working.reset_index(drop=True), target=target,
        summary={'total':len(working), 'train':len(train_idx), 'validation':len(val_idx), 'test':len(test_idx),
                 'normalize':normalize, 'imputation':impute, 'seed':seed,
                 'train_fraction':train_fraction, 'validation_fraction':val_fraction,
                 'test_fraction':1-train_fraction-val_fraction},
    )
