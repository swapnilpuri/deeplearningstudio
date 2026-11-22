# M11 acceptance criteria and review map

Requirements are preserved in [M11_REQUIREMENTS.md](M11_REQUIREMENTS.md).
Implementation decisions are in [M11_DESIGN.md](M11_DESIGN.md); exact test
outcomes and unverified checks are in [M11_COMPLETION_REPORT.md](M11_COMPLETION_REPORT.md).

| Requirement | Delivered surface / automated acceptance |
| --- | --- |
| 11.1 Overview | Concepts & Math family comparison, actual model diagram and stage selector |
| 11.2 Positions | Fixed M8 sinusoidal + learned tables, seed/dimension/length controls, numerical and heatmap inspection; deterministic/value/gradient/bounds tests |
| 11.3 Encoder layer | M10 self-attention/FFN, bidirectional + source PAD, ReLU/GELU, both norm orders, branch/input dropout; float64 PyTorch parity |
| 11.4 Encoder stack | 1–4 independently initialized layers, layer selection, comparison delta, norms and memory; multi-layer tests |
| 11.5 Decoder layer | Causal self-attention → cross → FFN with three residual/norm steps; stage inspection and reference parity |
| 11.6 Cross-attention | Decoder Q / memory K,V; M9 arithmetic, scores/mask/probabilities/value contributions/concat/output; explicit projection-origin tests |
| 11.7 Cross visualizations | Rectangular source-key/target-query heatmaps, decoder layer/head comparison, selected numerical cell, fixed probability scale, trained example selection |
| 11.8 Model builder | Separate source/target vocabularies, positions, dimensions, layer counts, FFN, norm, activation, dropout and seed; hierarchy and real counts |
| 11.9 Masks | Decoder mask explorer switches encoder PAD / causal+target PAD / cross source PAD; allowed positions, logits and probabilities; fully masked rows rejected |
| 11.10 Seq2seq task | Offline deterministic copy default / reversal optional; separate PAD/BOS/EOS vocabulary and padded variable-length examples |
| 11.11 Teacher forcing | Strict target shifting, separate encoder/decoder IDs, logits and per-position labels/loss; PAD CE ignored and causality tested |
| 11.12 Inference | Encoder computed once, no-grad greedy prefix recomputation, EOS per batch member, PAD/BOS suppression, length stop and mode restoration tested |
| 11.13 Training | Explicit action, Adam/SGD, bounded steps/batch/config, loss and accuracy/gradient graphs, before/after greedy examples, reset; measured default learning test |
| 11.14 Tensor inspector | Source/target lookup and positions, each layer's attention/residual/FFN tensors, memory, logits and probabilities with shape/sample/operation |
| 11.15 Architecture diagram | Configuration-driven encoder/decoder clusters, memory K/V edges, correct pre/post norm and residuals, dropout and projection; stage selection beside diagram |
| 11.16 Comparison | Encoder-only / decoder-only / encoder-decoder masks, sequences, objectives, cross-attention and outputs table |
| 11.17 Counts/complexity | Actual disjoint trainable categories and total; H*S²,H*T²,H*T*S score cells and QK/AV MAC estimates; worked count test |
| 11.18 Quiz/exercises | Existing question tuple and progress v1 conventions, 15 checks with feedback, 8 guided experiments, saved/draft answers/notes and completion/export/import |
| Navigation | Direct shared horizontal component, lazy open panels, m11-prefixed keys, session-persistent inputs, no auto training; existing navigation suite plus M11 AppTest |
| Compatibility | Additive learning path and files; no changes to M1–M10 mathematics, existing navigation or progress schema; full regression required |

## Browser/manual acceptance procedure

1. Run `python -m streamlit run app.py` with the project `.venv`; choose
   Learning Studio → Transformer Encoder & Decoder. Compare the seven tabs with
   the M10 tabs: active indication, spacing, overflow, keyboard and narrow width.
2. Open every tab. Change configuration in Concepts; switch away and back.
   Confirm controls retain values and only the active lazy panel is populated.
3. Try positions of length 3, D=4 (sinusoidal and learned). Confirm the position-0
   sinusoid is `[0,1,0,1]`. Select two encoder layers and compare representations.
4. Set S=5 and T=3; inspect decoder masking and rectangular cross heatmaps. Select
   a padded source key and verify probability zero. Inspect the numeric cell and
   weighted-value contributions; compare another layer/head on the same scale.
5. Restore defaults; train the copy experiment. Inspect decreasing loss, before/
   after predictions, teacher-forced and greedy accuracies, gradient curve and
   trained cross-attention. Navigation must not train again. Change steps; saved
   results remain labeled with their original configuration. Reset removes them.
6. Grade the quiz, leave draft notes, navigate away/back and verify preservation.
   Save notes and mark complete. Export progress and import it in a fresh session;
   verify both M11 and earlier lesson records. Check download dialogs separately.
7. Exercise earlier M1–M10 workspaces. Automated coverage does not by itself prove
   browser appearance, keyboard interaction, file dialogs or educational acceptance.

No manual check should be marked passed unless actually executed. Independent
educational review and optional GPU validation remain separate from unit/runtime
acceptance.
