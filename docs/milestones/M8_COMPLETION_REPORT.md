# M8 completion report

Date: 2026-11-21. Starting commit: dd878cac84c789ce14c17eb6061e32ec220baf6a.
Clean master contained the M7 merge and matched remote master. Branch:
feature/m8-embedding-learning-studio. No user changes overwritten.

## Implemented

Twelve-section Embedding Network lesson with 22 concepts, worked numerical
examples, equations, dimensions, real PyTorch demonstrations and guided experiments.
Word/character tokenization, corpus/toy vocabulary, PAD/UNK, frequencies, padding;
configurable real embedding table, custom JSON, one-hot lookup agreement, shapes,
rows and storage distinctions; gradients, repeated occurrences, padding/unused rows,
central differences, exact SGD and actual Adam updates.

Editable dot/cosine/Euclidean/Manhattan comparison with safe zero handling; exact
normalized or raw retrieval and full cosine matrix. Seeded offline skip-gram with
two trainable tables, stable log-sigmoid loss and eligible negative sampling;
initial/trained neighbors, direct 2D/3D, joint PCA and optional validated t-SNE.
Masked mean/sum/max sentence pooling, empty/all-PAD handling and reversal examples.
Learned and sinusoidal positions including odd dimensions, position comparisons,
token reversal and editable static cool. comparison with correct context limits.

Framework-neutral serialized schema v5 for embedding/optional positions/pooling/
linear logits with inferred shapes and counts; all prior schemas unchanged.
Offline supervised embedding classifier with train-only vocabulary, disjoint raw
splits, mini-batches, CE, SGD/Adam, train/validation metrics and explicit test
metrics/confusion/sample predictions. Last-batch gradients and initial/trained rows.
Twenty-one quizzes and shared progress v1 including actual headless legacy JSON
import preserve historical completion, notes and scores. Volume 6, README, roadmap,
ADR-0004 and all five M8 milestone records. No new dependencies or downloads.

## Actual verification

Existing Windows Python 3.12 environment, PyTorch CPU, Streamlit 1.67.0.

| Check | Executed result |
|---|---|
| Restricted baseline full pytest | 180 passed, 1 ONNX temp-fixture permission error, 32.65s |
| Normal-permission baseline ONNX retry | 1 passed, 42.23s |
| Embedding numerical/contract tests | 54 passed, 13.55s |
| Initial embedding UI run | 17 passed, 1 assertion on intentional limitation warning; changed notice to informational |
| Corrected embedding UI workflows | 18 passed, 18.98s |
| Final full automated suite | **253 passed, 59.89s; no failures/errors/skips** |
| Edited-source ty | Passed, including learning/ui.py |
| Compileall and git diff --check | Passed |
| Actual localhost Streamlit | 32 ready snapshots, twelve M8 sections plus old lessons; no new console warnings/tracebacks |

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_embedding.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_embedding_ui.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
uv tool run ty check --python .venv learning/embedding_spec.py learning/embedding_adapter.py learning/embedding_training.py learning/embedding_ui.py learning/embedding_content.py visualization/embedding_graphs.py learning/ui.py
.\.venv\Scripts\python.exe -m compileall -q learning visualization tests
git diff --check
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8768 --server.enableAgentApi=true
```

Normal permissions resolved Windows temp and existing uv-cache restrictions;
no assertions weakened or tests disabled. Initial cold runtime request timed out;
collecting the same session returned ready. Real-server interactions exercised
all sections, actual one-epoch word training, t-SNE and 3D PCA, empty max pooling,
sinusoidal position reversal, learned-position/max builder (106 parameters),
one-epoch classifier and explicit held-out test (194 parameters, CE **0.762522**,
accuracy **0.470588**), all 21 quiz answers/grade, notes/completion, and routing back
to feed-forward/deep/CNN/RNN/LSTM. These measurements establish workflows, not quality.

Separate seed sensitivity demonstration: one epoch, D=8, batch=256, Adam .03,
seeds 42 and 43. Train loss 4.542853 versus 5.205765; fixed-negative objective
4.008629 versus 4.616802. Cat's top-five cosine neighbors shared car/orange but
otherwise differed. Both trained matrices were finite. This is illustrative
initialization sensitivity, not a multi-seed quality study.

## Boundaries and outstanding acceptance

Passed: numerical/gradient/optimizer/serialization/count/padding/progress/split/
sampling/reproducibility/training tests, all earlier regressions including ONNX,
headless UI and actual running-app interactions. Failed: no remaining known source
or test failure. Skipped: none. Environment issues resolved as above.

Browser attempt used the computer-use plugin. It inspected the dedicated Edge
local studio at port 8767 (the earlier M7 session). Address replacement timed out;
reobservation showed the old URL unchanged. Enumeration/state capture recovered,
but keyboard input then failed with "failed to activate captured window". No M8
browser rendering, chart hover, editor interaction, file roundtrip or download
verification pass is claimed. Existing user server/session was not stopped.
Fresh install, Netron and independent architecture/educational acceptance remain
unexecuted. Browser helper failure is not evidence of an application defect.

Scope: transparent whitespace/character tokenization, static toy embeddings and
simple pooling, uniform-negative skip-gram, tiny generated binary classification;
no production tokenizer, pretrained encoder, attention, Transformer, vector index,
RAG, new dependency or embedding ONNX export. All prior ONNX tests passed.

**Final acceptance: implementation and automated/runtime verification passed;
manual/browser and independent acceptance pending.** See M8_ACCEPTANCE_CRITERIA.md.

## Git delivery

Implementation commit: `f2d76a3cf90805d6bbe830ead58cff5e5e9a04d8`, pushed on
`feature/m8-embedding-learning-studio` and verified against the remote branch.
[Draft PR #5](https://github.com/swapnilpuri/deeplearningstudio/pull/5) targets
`master`; verified open/draft with the correct head commit. No merge performed.

The GitHub connector returned 403 (resource not accessible by integration) for PR
creation. The first GitHub CLI attempt was not executed because automatic approval
review hit an account usage limit. After the user requested resumption, the same
approved CLI workflow created the PR successfully. No approval bypass was used.

This delivery follow-up changes documentation only; source remains exactly as
tested in the 253-test regression, so tests were not repeated. Final containing
documentation commit is resolved by `git rev-parse HEAD`. Browser/manual and
independent acceptance remain pending as recorded above.
