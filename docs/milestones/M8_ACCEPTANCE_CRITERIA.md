# M8 acceptance evidence

Date: 2026-11-21. Implementation, automated/runtime verification and core browser
acceptance passed. Independent acceptance remains pending; this is not an
independent acceptance sign-off. Browser evidence/coverage limits are recorded
in M8_COMPLETION_REPORT.md under Dedicated browser acceptance.

| Required outcome | Implementation / evidence | Status |
|---|---|---|
| Preserve and extend Learning Studio | learning/ui.py routing, all prior regression and real-server lesson selection | Passed |
| Beginner explanations, math, shapes, examples | 22 content topics with explorer mapping; Volume 6 tutorials | Implemented; independent educational review pending |
| Tokenization/vocabulary/PAD/UNK/frequencies | embedding_adapter.py, Tokenization & vocabulary, numerical and AppTest cases | Passed |
| One-hot/direct lookup equality | real nn.Embedding, multiplication equality tests and live max-error display | Passed |
| Configurable table/rows/shapes/counts | Matrix & one-hot controls including custom JSON; validation tests | Passed |
| Gradients/repeated rows/PAD/SGD/Adam | double-precision central difference, occurrence accumulation, exact optimizer tests | Passed |
| Four similarity metrics/zero/normalization | numerical tests, editable vectors, real-server checks | Passed |
| Exact Top-K retrieval | original-space cosine/dot, normalization, stable ranking tests | Passed |
| Offline word learning | stable log-sigmoid skip-gram, seeded negatives excluding known positives, SGD/Adam smoke | Passed |
| Initial/trained vector visualization | direct 2D/3D, joint PCA, optional validated t-SNE; real-server t-SNE; browser 2D/3D PCA inspected | Passed; Direct/t-SNE browser variants unexecuted |
| Masked sentence pooling | mean/sum/max, empty/all-PAD/negative max and order-invariance tests | Passed |
| Position representations | real learned table, correct odd/even sinusoidal formulas, reversal controls | Passed |
| Static/contextual distinction | shared cool. row in editable sentences, no false attention/context claim | Implemented; independent review pending |
| Architecture builder and neutral specs | v5 JSON roundtrip, shape propagation, count/actual parameter agreement, no torch in spec | Passed |
| Downstream classifier | offline mini-batches, CE, SGD/Adam, train/val metrics, explicit test/confusion/predictions | Passed |
| Leakage prevention | disjoint unique raw records; vocabulary fit on train; held-out label poison does not affect training | Passed |
| Evaluation and limitations | neighbor changes, projections, retrieval, test metrics, guided seed repeats; separate two-seed executed experiment | Passed workflow; no semantic quality claim |
| Quizzes/progress backward compatibility | AppTest legacy JSON preserves M4–M7; browser 21/21 quiz, saved notes/completion and export/upload/load/re-export exact JSON equality | Passed; legacy import verified headlessly |
| Dense/CNN/RNN/LSTM, datasets/training/ONNX | complete regression 253 passed, no skips/failures | Passed |
| Volume 6, README, roadmap, ADR and milestone records | All created/updated; Volumes 1–5 unchanged | Implemented |
| Commit/push/PR against master | f2d76a3 pushed and verified; draft PR #5 targets master, open/unmerged | Passed |
| Manual/browser and independent acceptance | All twelve sections, both training labs, custom JSON, positional/builder controls, PCA plots, full quiz and downloads/progress roundtrip exercised in dedicated Edge localhost:8501 | Core browser passed; independent review pending |

Optional BPE, analogies, external semantic search, pretrained models, vector indexes
and attention/RAG are not implemented. Optional t-SNE is implemented. Embedding
ONNX export is outside this milestone's applicable existing ONNX checks.

Next: independent architecture and educational review from committed source and
these records. Optional expanded browser checks: deliberate hover/rotation, Direct
2D/3D and t-SNE plots, every invalid-input/control variant and legacy progress JSON
import. Core navigation, rendering, custom JSON, both training paths, full quiz,
specification download and progress file roundtrip have now passed in the browser.
