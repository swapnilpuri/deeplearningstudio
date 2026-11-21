# M8 acceptance evidence

Date: 2026-11-21. Implementation and automated/runtime verification passed;
browser/manual and independent acceptance remain pending. This is not an
independent acceptance sign-off.

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
| Initial/trained vector visualization | direct 2D/3D, joint PCA, optional validated t-SNE; real-server t-SNE and 3D PCA | Passed runtime; visual inspection pending |
| Masked sentence pooling | mean/sum/max, empty/all-PAD/negative max and order-invariance tests | Passed |
| Position representations | real learned table, correct odd/even sinusoidal formulas, reversal controls | Passed |
| Static/contextual distinction | shared cool. row in editable sentences, no false attention/context claim | Implemented; independent review pending |
| Architecture builder and neutral specs | v5 JSON roundtrip, shape propagation, count/actual parameter agreement, no torch in spec | Passed |
| Downstream classifier | offline mini-batches, CE, SGD/Adam, train/val metrics, explicit test/confusion/predictions | Passed |
| Leakage prevention | disjoint unique raw records; vocabulary fit on train; held-out label poison does not affect training | Passed |
| Evaluation and limitations | neighbor changes, projections, retrieval, test metrics, guided seed repeats; separate two-seed executed experiment | Passed workflow; no semantic quality claim |
| Quizzes/progress backward compatibility | 21 questions, notes, completion; actual AppTest legacy JSON upload/load preserving M4–M7 records | Passed headless/runtime; browser roundtrip pending |
| Dense/CNN/RNN/LSTM, datasets/training/ONNX | complete regression 253 passed, no skips/failures | Passed |
| Volume 6, README, roadmap, ADR and milestone records | All created/updated; Volumes 1–5 unchanged | Implemented |
| Commit/push/PR against master | Delivery recorded in completion/handoff after execution | See delivery records |
| Manual/browser and independent acceptance | Browser helper could inspect the old local app but navigation/activation failed; no M8 rendering pass claimed | Unverified / pending |

Optional BPE, analogies, external semantic search, pretrained models, vector indexes
and attention/RAG are not implemented. Optional t-SNE is implemented. Embedding
ONNX export is outside this milestone's applicable existing ONNX checks.

Resume acceptance: start the app, select Embedding Network and inspect each of
the twelve sections in a functioning browser. Check chart hover/layout, custom
JSON editing, Direct 2D/3D plots, both training paths, actual download/upload
roundtrip, specification download and full quiz submission. Perform independent
architecture and educational review from the committed source and these records.
