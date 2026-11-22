# M8 handoff

Status: implementation, automated/runtime verification and core browser acceptance
passed; independent acceptance pending. Date: 2026-11-21.
Branch: feature/m8-embedding-learning-studio.
Starting commit: dd878cac84c789ce14c17eb6061e32ec220baf6a, clean master,
M7 merge confirmed; remote master matched via git ls-remote deeplearningstudio.
Current containing commit: resolve git rev-parse HEAD (a file cannot name its own
containing commit). Implementation commit: f2d76a3cf90805d6bbe830ead58cff5e5e9a04d8.
Pushed and verified against the remote feature branch. [Draft PR #5](https://github.com/swapnilpuri/deeplearningstudio/pull/5)
targets master; open/draft, correct head verified, no merge. Connector creation
returned 403; CLI creation succeeded after user-requested resumption of an initial
approval-review usage-limit interruption. No approval bypass. This final follow-up
is documentation only; no source changes after the 253-test regression.

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

The earlier port-8767 browser limitation was superseded by user-authorized browser
acceptance in dedicated single-tab Edge at http://localhost:8501 on 2026-11-21,
dark theme, 80% zoom. All twelve sections exercised with native UI/screenshot
inspection. Verified word/character tokenizer, custom matrix JSON and exact lookup
equality, SGD/Adam and finite-difference display, similarity inputs/normalization,
five-epoch skip-gram and initial/trained 2D/3D PCA, empty max sentence pooling,
learned/sinusoidal D=5 position modes and reversal, builder shape graph/counts and
learned-position increase 58→106, actual schema-v5 spec download, five-epoch
classifier training and explicit held-out test (CE 0.596323, accuracy 0.647059,
confusion [[4,3],[3,7]] and predictions), evaluation screen, full 21/21 quiz,
notes, completion and actual browser progress export/upload/load/re-export.
Read back task downloads: spec matches controls; progress JSON before/after import
is exactly equal with score, completion and saved notes intact. No app defect
identified; user server remains running. Details: M8_COMPLETION_REPORT.md.

Browser coverage limits: Direct-mode plots, t-SNE, deliberate hover/rotation,
all invalid-input/control variants and legacy M4–M7 JSON import were not exercised
in this browser session (applicable numerical/headless/runtime checks already
passed). Fresh install, Netron and independent review remain unexecuted. Native
automation needed reobservation/scrolling and keyboard selection for clipped menu
options; no source fix was required. No new code since the 253-test regression.

Highest-priority next task: independent architecture/education acceptance and PR
review; optionally extend browser variants listed above. Do not merge automatically.
Implementation and core browser acceptance passed; milestone sign-off still awaits
independent review. See M8_ACCEPTANCE_CRITERIA.md.

Resume commands (repository root):
`.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_embedding.py`;
`.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_embedding_ui.py`;
`.\.venv\Scripts\python.exe -m streamlit run app.py`;
`git status --short`; `git rev-parse HEAD`.
