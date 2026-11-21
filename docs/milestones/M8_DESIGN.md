# M8 design

Use separate embedding schema v5; dense v1, CNN v2, RNN v3, LSTM v4 and progress
v1 remain unchanged. EmbeddingModelSpec validates dimensions, padding, pooling,
positional capacity, serialized identity, shapes and parameter counts without
framework imports. The PyTorch adapter executes lookup, masked pooling, optional
position addition and a linear head. No attention or contextual encoder.

Tokenization and vocabulary fitting precede numerical computation. PAD=0/UNK=1
are stable corpus vocabulary conventions; the architecture permits any valid pad
index or no padding. Custom embedding rows are explicit values, including PAD;
padding_idx suppresses gradients rather than forcibly overwriting custom values.

Skip-gram uses two real embedding tables and stable log-sigmoid negative sampling.
Uniform negative candidates exclude special tokens, target and all known positive
contexts for that target. No eligible negative produces a clear validation error.
Classification splits unique records first and fits vocabulary only on train;
training reads train/validation only, with explicit held-out evaluation.

All math and training live outside Streamlit. Plotly helpers take arrays and return
figures. Expensive training occurs only on explicit buttons; configuration changes
invalidate stored results. Progress uses the existing schema and shared record.
See [ADR-0004](../adr/ADR-0004-embedding-representations.md).
