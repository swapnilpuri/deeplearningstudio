# M9 design

Base: merged M8 master d56d05d. Existing navigation, embedding adapters, visualization
helpers and progress schema remain authoritative. No new dependencies or serialized
architecture version were needed: this milestone is a computation laboratory,
not a general Transformer builder. No consequential architecture change or ADR.

| Layer | Source | Responsibility |
|---|---|---|
| Validation | learning/attention_spec.py | frozen width/seed/initialization specification |
| Math | learning/attention_adapter.py | projections, visibility, traceable differentiable attention |
| Experiment | learning/attention_training.py | validated config, deterministic routing/autograd/updates |
| Visualization | visualization/attention_graphs.py | query/key heatmaps, actual routing bars |
| Presentation | learning/attention_ui.py | controls, tables, lesson routing, explicit actions |
| Education | learning/attention_content.py | nine explanations, quiz and answer checking |
| Shared | learning/ui.py, progress.py, embedding_adapter.py | lesson registry, existing progress and embeddings |

Weights are explicit [D,dk/dv] matrices; no hidden Linear transpose or bias.
Q/K widths agree, V width may differ; [T,D] input gives [T,dv] output.
Mathematical core preserves autograd, validates finite inputs, checks overflow,
uses stable torch.softmax and returns all stages. True visibility means allowed.
Causal mask is lower triangular. Padding blocks keys only; padded query rows are
explicitly identified. Every query must retain a key; fully masked rows raise
ValueError. -infinity is intentional in masked logits only. Public direct Q/K/V
attention can support cross-length queries later, while M9 UI remains self-attention.

M8 learned mode reads the existing emb_word_run trained table/vocabulary at its
actual width. Synthetic mode uses the same lookup construction so duplicate tokens
have consistent static rows. No duplicate tokenizer or embedding training.

Training uses separate fixed identity X [4,4], Wq/Wk [4,3], Wv [4,2]. Cyclic routing
cross entropy plus output MSE makes all three projections learnable. Each explicit
Train starts fresh. Session result is invalidated on config change; Reset discards
it and displays regenerated identical initial weights. Computation has no Streamlit
imports. Namespaced att_ widget keys isolate it from M8/legacy workspaces.

UI bounds: 24 tokens, widths <=16 (M8/core <=32), <=500 training steps. Seeds use
local generator or existing M8 RNG isolation. Existing Plotly curves and progress v1
are reused. See Volume 7 for numerical worked example and interpretation limits.
