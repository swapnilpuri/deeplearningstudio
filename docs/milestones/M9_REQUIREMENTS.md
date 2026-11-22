# M9 requirements and scope

Implement an offline, CPU-first single-head self-attention lesson incrementally
from merged M8 master d56d05d67f4dade5cb9472244b891b33e49c75d5.
The user-supplied M9 specification is mapped below; code and acceptance records
are the durable source of implementation status.

1. Register nine progressive sections in existing Learning Studio navigation.
2. Reuse M8 tokenizer, vocabulary, seeded embedding table and session-trained weights;
   support synthetic and user JSON inputs, dimensions, token IDs and tensor values.
3. Show editable bias-free row-vector Q/K/V projections and shape inspection.
4. Inspect raw/scaled scores, dot products, key-axis softmax, normalized probabilities,
   masked logits, mask visibility and contextual weighted value outputs.
5. Provide labeled interactive Plotly heatmaps, query/key and score-cell inspection,
   selected-query routing distributions and input/output comparison.
6. Support bidirectional, causal, padding and combined visibility with explicit
   fully masked row rejection and zero forbidden probabilities.
7. Train a deterministic synthetic routing task with autograd, SGD/Adam, explicit
   action/reset, loss, gradients, parameter deltas and before/after routing.
8. Check nine guided exercises; reuse progress JSON schema v1 for scores/notes/completion.
9. Verify math, masks, determinism, gradients, updates, convergence of default task,
   UI validation/state, M8 reuse and full M1–M8 regression.
10. Deliver Volume 7, design/acceptance/evidence/handoff, updated roadmap and a
    reviewable feature branch/PR; never merge master automatically.

Excluded: multi-head attention, Transformer blocks, GPT, KV caches, optimized
attention kernels, external APIs/services/downloads, new backend/storage/frameworks.
See [design](M9_DESIGN.md), [acceptance](M9_ACCEPTANCE_CRITERIA.md),
[learning guide](../VOLUME_7_SELF_ATTENTION.md).
