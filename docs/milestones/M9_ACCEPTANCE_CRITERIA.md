# M9 acceptance criteria

Implementation and automated/runtime validation completed on 2026-11-21.
Independent educational/architecture acceptance and browser visual QA remain
unexecuted; these are separate from the implementation checks below.

| Requirement | Status | Evidence |
|---|---|---|
| M9.1 nine progressive sections, existing navigation | Passed | test_attention_ui.py sections; real-server all nine |
| M9.2 token input/IDs/shapes, synthetic, M8, learned/custom | Passed | M8 learned-table equality workflow, custom validation, seeded core |
| M9.3 editable Q/K/V, unequal dk/dv, no bias | Passed | hand matrix test, manual UI workflow, displayed matrices |
| M9.4 scores and selected dot-product calculation | Passed | hand test, query/key controls, selectable score table |
| M9.5 scaled/raw comparison and scaling explanation | Passed | hand test, live section, paired softmax tables |
| M9.6 key-axis softmax, row sums, stable probabilities | Passed | normalization/extreme-logit tests, probability heatmap/bars |
| M9.7 weighted outputs/input comparison | Passed | hand test and live weighted-contribution/output displays |
| M9.8 causal/padding/combined masks and blocked rows | Passed | exact masks/zero probabilities/invalid rows tests, live causal |
| M9.9 raw/scaled/probability/mask views | Passed | all sections exercised; plotting preparation in attention_graphs.py |
| M9.10 selected-query actual routing distribution | Passed | real-server chart and shared probability trace |
| M9.11 explicit autograd SGD/Adam, reset, weight/gradient/loss views | Passed | deterministic training tests, real-server Train and Reset |
| M9.12 tensor computation inspector | Passed | every numerical section displays shapes/dtypes/ops/samples |
| M9.13 nine checked exercises/shared progress | Passed | quiz/notes/completion/export roundtrip, legacy import preservation |
| M9.14 Volume 7 and milestone records | Passed | linked requirements/design/report/handoff and roadmap |
| Mathematical and numerical conditions | Passed | 14 core tests incl. PyTorch SDPA reference, all finite outputs |
| M1–M8 regression | Passed | full suite 281 passed; no skips/failures |
| Edited-source type and syntax checks | Passed | ty, compileall, git diff --check |
| Real running application | Passed | 23 ready agent-API snapshots without warnings/exceptions |
| Browser-only score-table click/layout | Unverified | API marks dataframe selection action_unsupported; selectors work |
| Independent milestone acceptance | Pending | external architecture/educational review not performed |

Scope limits: no production semantics claim, positional encoder, generic attention
architecture serialization, export, multi-head/Transformer blocks or large models.
The optional directed graph is represented by an actual query-to-key probability
bar distribution; no additional graph package. Heatmaps support hover/zoom and the
score table supports cell selection; query/key selectors provide the always-available
inspection path. See [completion report](M9_COMPLETION_REPORT.md).
