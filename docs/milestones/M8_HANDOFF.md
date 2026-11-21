# M8 handoff

Status: implementation and automated/runtime verification passed;
browser/manual and independent acceptance pending. Date: 2026-11-21.
Branch: feature/m8-embedding-learning-studio.
Starting commit: dd878cac84c789ce14c17eb6061e32ec220baf6a, clean master,
M7 merge confirmed; remote master matched via git ls-remote deeplearningstudio.
Current containing commit: resolve git rev-parse HEAD (a file cannot name its own
containing commit). Git delivery details appended after execution.

References: M8_REQUIREMENTS.md, M8_DESIGN.md, M8_ACCEPTANCE_CRITERIA.md,
../adr/ADR-0004-embedding-representations.md.

Baseline restricted pytest: 180 passed, 1 ONNX temp-fixture permission error in
32.65s. Normal Windows ONNX retry: 1 passed in 42.23s. No tests disabled.

Implemented twelve sections, 22 concepts, real embedding matrix/one-hot lookup,
SGD/Adam/finite-difference gradients, four metrics, exact retrieval, deterministic
skip-gram, direct 2D/3D/PCA/t-SNE, masked pooling, learned/sinusoidal positions,
static/context distinction, neutral schema v5 classifier builder, supervised text
training and explicit held-out test evaluation. Added 21 quizzes, guided exercises,
legacy-compatible progress, Volume 6, ADR and milestone records. No new dependency.

Files: learning/embedding_{spec,adapter,training,content,ui}.py;
visualization/embedding_graphs.py; tests/test_embedding{,_ui}.py; learning/ui.py;
README, roadmap, Volume 6, ADR-0004 and the five M8 milestone records.
Dense v1/CNN v2/RNN v3/LSTM v4/progress v1 unchanged; Volumes 1–5 preserved.

Verification: numerical 54 passed in 13.55s; headless UI 18 passed in 18.98s;
final complete regression 253 passed in 59.89s, no failures/errors/skips.
Edited-source ty, compileall and whitespace checks passed. Real server port 8768:
32 ready snapshots including initial collection and 31 subsequent transitions,
all twelve M8 sections, one-epoch skip-gram, t-SNE, 3D PCA, empty max pooling,
sinusoidal reversal, learned-position builder, classifier train/test, 21/21 quiz,
notes/completion and all five former lesson routes. Console had no new warnings
or tracebacks. One-epoch classifier test CE 0.762522, accuracy 0.470588;
workflow evidence only. Separate one-epoch seeds 42/43 changed cat neighbors and
both produced finite embeddings; no general semantic/stability claim.

Browser helper inspection confirmed the dedicated Edge window still showed the
older local app on port 8767. Setting its address timed out; reobservation confirmed
no navigation occurred. Recovery enumeration/state capture succeeded after retries,
but keyboard action failed to activate the captured window. No M8 browser rendering
or browser interaction pass is claimed. The user's existing server was not stopped.
No known failing source test; visual/manual review, file roundtrip in browser,
fresh install, Netron and independent review remain unexecuted.

Highest-priority next task: browser/manual M8 acceptance with a working browser
automation connection, then independent architecture/education review and PR review.
Do not merge automatically. Implementation is verified, milestone acceptance is
conditional on those outstanding checks. See M8_ACCEPTANCE_CRITERIA.md.

Resume commands (repository root):
`.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_embedding.py`;
`.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_embedding_ui.py`;
`.\.venv\Scripts\python.exe -m streamlit run app.py`;
`git status --short`; `git rev-parse HEAD`.
