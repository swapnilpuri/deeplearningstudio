# ADR-0004: Embedding representations

Status: accepted for implementation, 2026-11-21.

Add a bounded framework-neutral embedding classifier specification, schema v5,
alongside the previous dedicated schemas. Reuse shared lesson routing and progress
v1 without migrating historical definitions. Represent positional mode and pooling
as validated fields rather than introducing a general graph/operator system.

Real execution belongs to EmbeddingAdapter. Token lookup, learned/sinusoidal
position addition, masked pooling and linear logits provide the M8 building blocks
for future attention. Position addition does not mix information across tokens.
Mean/sum pooling of token-plus-position vectors can still lose token-order
information; position addition alone is not a contextual encoder.

Use exact vector search for the small offline corpora. No pretrained downloads,
external vector databases, new dependencies, attention or RAG are required.
