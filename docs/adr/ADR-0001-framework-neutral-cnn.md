# ADR-0001: Introduce CNN specifications alongside dense ModelSpec

**Date:** 2026-11-21  
**Status:** Implemented and regression verified

## Context

M4 `ModelSpec` serializes dense neural networks and is used by legacy tests and the PyTorch adapter. M5 adds spatial operations and output-shape inference. Mutating dense `LayerSpec` into a universal union would risk breaking v4 serialization and tests.

## Decision

Add `CNNModelSpec` and `CNNLayerSpec` in `learning/cnn_spec.py`, with explicit `schema_version=2` and `architecture='cnn'`. Keep M4 `learning/spec.py` and its JSON unchanged. Shape propagation accepts NCHW input, tracks convolution, activation, pooling, flatten and dense operators, rejects invalid sequences, computes weight shapes and parameter counts. Implement CNN PyTorch execution independently in `learning/cnn_adapter.py`. Retain portable progress schema version 1, allowing the new `cnn` lesson identifier without data loss.

## Consequences

The CNN shape engine is backend-neutral and serializable but only covers standard ungrouped convolutions and zero-padded, non-dilated pooling. No TensorFlow runtime is introduced. A future model-neutral operator registry can unify dense, CNN, embedding and attention operators with an explicit migration, rather than silently changing old artifacts.

## Final verification

Rectangular/dilated convolution and hidden dense heads use the same shape engine; rows include input/output, weight/bias shapes and parameter counts. Actual PyTorch forward traces verify propagation. M4 JSON/progress compatibility has regression coverage. CNN training uses forked model initialization and its own seeded DataLoader generator, avoiding process-wide seed mutation.
