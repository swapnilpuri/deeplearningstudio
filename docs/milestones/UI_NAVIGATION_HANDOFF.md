# Learning-section navigation handoff

- Status: implementation and automated validation complete; delivery results below.
  Browser visual acceptance remains pending because the browser access host is unavailable.
- Branch: `feature/ui-consistent-learning-navigation`; target: `master`.
- Initial local baseline: `fccd232f04cebc825b52e3d2e3f85679104a08c9` on the
  clean M10 integration branch. Fetch confirmed M10 PR #7 was merged. The feature
  branch was fast-forwarded to `3b062d75aa94791454cd74c4f27ae85e7247a001`
  (current master; identical source tree) without discarding local edits.
- Current implementation commit: see `git log -1 --oneline` on this branch.
  No merge into master is authorized or performed.

## Inventory and implementation

| Inspected studio/workflow | Change |
| --- | --- |
| Neural Network Studio | Existing four native tabs now use shared wrapper; eager rendering retained |
| Feed-Forward foundations | Existing six native tabs now use shared wrapper; eager rendering retained |
| Deep Neural Network foundations | Same foundation renderer and labels; eager rendering retained |
| General Training Studio | Existing five native tabs now use shared wrapper; eager rendering retained |
| Dataset Management | Single preparation workflow; source/dataset selectors are input controls, not educational section navigation |
| CNN | Seven section dropdown options replaced with native tabs |
| RNN | Nine section dropdown options replaced with native tabs |
| LSTM | Nine section dropdown options replaced with native tabs |
| Embedding (M8) | Twelve section dropdown options replaced with native tabs |
| Self-Attention (M9) | Nine section dropdown options replaced with native tabs |
| Transformer Blocks (M10) | Twelve section dropdown options replaced with native tabs |

`learning/navigation.py::render_learning_section_navigation` accepts ordered
labels or an ID-to-label mapping and returns ordered ID-to-tab containers.
Keys are `<module_key>_section`; existing lazy studio keys and labels are retained.
Native `st.tabs(on_change='rerun')` plus `tab.open` guards preserve single-panel
execution. Inactive panels do not compute traces, render quizzes, or train models.
Section inputs use their existing stable keys with `persist_state='session'`.
Button, upload and editor state is not reassigned. Computation, training,
visualization functions, content and progress schema are unchanged.

The declared Streamlit minimum is now 1.67, matching the installed version and
the native APIs used for lazy tabs and input persistence. No package was installed
or upgraded. No custom HTML, CSS, JavaScript or frontend framework was introduced.
Native tabs own active styling, label layout and horizontal overflow.

`AGENTS.md` defines the mandatory shared convention for M11 and future modules.
There are no dropdown section-navigation exceptions. Eager rendering is retained
for the early studios because foundations have a forward-input/backward-panel
dependency and already compute their panels together. Workspace and learning-path
selectors stay separate from section navigation.

## Validation

Environment: Windows, Python 3.12.12, Streamlit 1.67.0, project `.venv`.

| Check | Result |
| --- | --- |
| Affected UI suites, including new navigation contracts | 121 passed in 83.99s |
| Final navigation contracts with training-call guards | 16 passed in 23.01s |
| Full M1–M10 regression (sequential final run) | 346 passed in 83.53s; no failures/skips |
| Edited source and navigation test `ty` | All checks passed |
| `compileall` and `git diff --check` | Passed |
| Live localhost agent API | 14 ready snapshots; six converted studios switched to their final exercise tabs; no warning/error/exception elements |
| Live server console | No warnings or tracebacks during successful checks |
| Desktop/narrow-width visual review | Not executed: browser access host unavailable |

New contracts cover ordered/accessibile tabs for every learning path, one populated
panel per lazy studio, selected-section state, independent module keys, distinct
IDs and labels, quiz answers and unsaved notes across tab switches, unchanged saved
progress, no training on navigation, and an explicit M10 training result/configuration
round trip with exactly one training call. Existing module suites retain their
training, exercise, data, computation and integration assertions.

Initial targeted run found two failures in a discarded state-preservation approach:
Streamlit prohibits assigning button state. Replaced it with native input persistence;
the corrected 121-test run passed. Initial type check caught an unavailable public
tab type export; removed that import and subsequent checks passed. Restricted uv
cache access failed, so the existing cached checker ran with normal permissions.
Two initial live probes needed corrections for timeout collection and formatted
selection labels; the final 14-snapshot traversal succeeded.

The first full regression run had 345 passes and one 30-second AppTest startup
timeout in `test_all_workspaces_render[Training Studio]` while live checks and
another test process were active. The timeout occurred on the initial default
workspace run. A sequential full rerun was started after stopping the live server;
the sequential rerun passed all 346 tests. No timeout or assertion was weakened.

Commands from repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_navigation.py tests/test_app.py tests/test_transformer_ui.py tests/test_rnn_ui.py tests/test_lstm_ui.py tests/test_embedding_ui.py tests/test_attention_ui.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
uv tool run ty check --python .venv learning/navigation.py learning/ui.py learning/cnn_ui.py learning/rnn_ui.py learning/lstm_ui.py learning/embedding_ui.py learning/attention_ui.py learning/transformer_ui.py app.py tests/test_navigation.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Manual visual acceptance

1. At a normal desktop width, open Neural Network Studio, Training Studio and each
   Learning Studio path. Confirm horizontal tabs, original labels/order and active
   styling. Select every section, including the final section of each long tab set.
2. Narrow the browser to roughly 390px. Scroll the native tab row (or use its native
   overflow controls) to the last tab; keyboard-focus and select it. Confirm no
   section is clipped or inaccessible. Repeat for the twelve-tab M8 and M10 studios.
3. Change an explorer input, leave its tab and return. Answer a quiz question and
   type unsaved notes; navigate away/back. Confirm the inputs and drafts remain.
4. Train once using a short nondefault configuration. Navigate to concepts and
   exercises, then return. Confirm configuration, curves, weights and test results
   remain and no training runs until the training button is explicitly pressed.
5. Save/complete an exercise, export progress, navigate, and import the JSON. Confirm
   earlier milestones' records remain. Verify file picker/download behavior manually.

Remaining limitations: visual/overflow and browser download dialog acceptance are
pending; fresh-install and optional GPU validation were not performed for this UI
refactor. No known calculation/training changes or functional failures.

## Delivery

Implementation and automated verification are ready for review. Next task:
independent review and the manual visual checks above; do not merge automatically.
Push and PR outcome will be recorded here after delivery.
