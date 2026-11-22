# M11 completion report

Date: 2026-11-22 (America/Chicago).

**Status: Partial — implementation and automated validation delivered; browser
visual/keyboard/download-dialog and independent educational acceptance pending.**

## Delivered

- Seven shared native horizontal tabs; lazy active-panel rendering, namespaced
  session-persistent controls, session-owned models/results, progress v1 reuse.
- Fixed sinusoidal and learned positions; inspectable encoder/decoder layers and
  stacks, both norm orders, activation/dropout configuration and padding/causal masks.
- Rectangular cross-attention using M9 arithmetic and M10 packed weights, Q/K/V
  provenance, masks, scores, probabilities, weighted values and layer/head comparison.
- Small complete encoder-decoder with separate vocabularies, hierarchy, actual
  parameter counts, theoretical attention-cost estimates and configuration-driven
  residual/normalization diagrams with tensor-stage selection.
- Offline variable-length copy/reverse examples, strict teacher-forcing shifts,
  PAD-aware cross-entropy, deterministic Adam/SGD training and no-grad greedy
  generation with one encoder call, EOS/length termination and per-sequence padding.
- Training loss, teacher-forced token/sequence accuracy, pre-clip gradient norms,
  before/after greedy predictions, selected trained attention, reset, fifteen
  graded questions, guided experiments, saved notes and portable learning progress.

The [acceptance map](M11_ACCEPTANCE_CRITERIA.md) links all 18 functional areas
to their surfaces/tests. See [design](M11_DESIGN.md) and
[learning guide](../VOLUME_9_ENCODER_DECODER.md).

## Baseline and compatibility

Started clean on `feature/ui-consistent-learning-navigation`; fetched the actual
remote `deeplearningstudio` and verified master `41e8cbfa6dd7d6cc9681dadc1693c4f356add0d7`
contains merged M10 PR #7 and shared navigation PR #8. Created
`feature/m11-transformer-encoder-decoder` from that remote master. Stale local
master was not rewritten. No pre-existing changes were discarded.

Only shared-source change is adding the new learning-path dispatch; the existing
navigation contract suite also includes the new module. No M1–M10 mathematical
implementation, navigation API, model schema or progress format was changed.
No new dependencies, services, datasets, credentials, model caches or binaries.

## Execution evidence

Environment: Windows, Python 3.12.12, PyTorch 2.15.0+cpu, Streamlit 1.67.0,
existing repository `.venv`.

| Check | Actual result |
| --- | --- |
| Initial positions/encoder/decoder numerical suite | 22 passed in 2.51s after correcting a test-only Tensor.cat typo |
| Initial mathematical + UI suite | 43 passed in 47.66s; process exit 0, then sandbox temp cleanup permission traceback |
| Expanded targeted suite under sandbox | 59 passed, 1 existing navigation test failed on Windows temp-file PermissionError; no assertion was weakened |
| Full regression, sequential with normal temp-file access | **394 passed in 144.33s; zero failures/skips** |
| Final targeted regression (M11 mathematics/UI + all navigation contracts) | **64 passed in 31.26s**, zero failures/skips |
| Final `ty` check of edited Python source and tests | All checks passed |
| `compileall` and `git diff --check` | Passed; Git warns that an existing test file will normalize LF to CRLF |
| Live Streamlit localhost agent API | **23 ready M11 snapshots**; final inspector rerun added 3 ready app/learning-path snapshots, no warning/error/exception elements |
| Live console | No warnings or tracebacks during successful runtime checks |
| Browser visual / keyboard / narrow width / dialogs | Not executed; browser host unavailable and unrelated-window selection rejected by automatic approval review |
| Optional GPU / fresh install / independent educational review | Not executed |

Full regression includes all existing M1–M10 tests. Final targeted verification
covers the small subsequent tensor-inspector input/output-shape clarification
and test type narrowing, without changes to mathematical behavior.

Reference parity: mapped custom row-vector Q/K/V/O to PyTorch encoder/decoder
layer projections, zeroed reference attention biases, copied FFN/norm parameters,
and compared float64 outputs and final projected logits at `atol=rtol=1e-9`.
Both norm orders and ReLU/GELU pass with dropout=0. Tests include batch padding,
unequal source/target lengths, future-target invariance, memory PAD invariance,
gradients, cross Q/K/V origin, real parameter formula, learned-position gradients,
invalid inputs, fully masked rejection, deterministic CPU dropout training,
EOS-per-sequence behavior, max-length stop and one encoder evaluation.

Live runtime traversal visited every M11 panel, entered/saved notes, graded all
15 questions, trained two explicit steps, generated greedily, changed sections
and returned to saved training, reset, and selected learned positions, post-norm
and two encoder/decoder layers. Initial navigation into Learning Studio and
default app creation were additional ready snapshots, outside the 23 M11 count.
The server was stopped before the sequential full regression run.
After the final tensor-inspector clarification, a separate three-snapshot live
check verified that the new input_shape/output_shape columns appear in the real
runtime. Its console was clean and that server was also stopped.

### Measured default learning

`train_seq2seq(Seq2SeqTrainingConfig())` — D=16,H=2,FF=32, one layer per stack,
fixed positions, pre-norm, no dropout, seed 42, 150 Adam updates, batch 8,
learning rate 0.01, 16 fixed copy examples (content lengths 1–3).

| Metric | Before | After |
| --- | --- | --- |
| Full training-set non-PAD cross-entropy | 2.2789463997 | 0.0154369855 |
| Greedy exact sequence accuracy | 0/16 | 16/16 |
| Final teacher-forced token accuracy | — | 100% |
| Final teacher-forced sequence accuracy | — | 100% |

Measured standalone elapsed time: 10.03 seconds. Final gradient norm before
clipping: 0.10547. These results demonstrate fitting the training set, not
generalization. Arbitrary seeds, large rates, reversal, SGD and short budgets
have no convergence guarantee. The default-learning test requires >10× loss
reduction, >=87.5% greedy exact match and >=95% token accuracy, rather than
hard-coding exact floating point values.

## Failures and resolutions

- Initial fetch used nonexistent `origin` and restricted Git could not create a
  signal pipe. Fetching the configured `deeplearningstudio` remote with normal
  permissions succeeded. No remote URL or credentials were changed.
- One initial position test used a nonexistent Tensor method; corrected the
  expected-value expression and all numerical tests passed.
- `ty` caught incompatible forward signatures inherited from self-attention/
  encoder classes. Reused M10 parameters by composition and factored residual
  behavior into a shared base with separate forward signatures. Added explicit
  buffer typing and narrowed test module/optional gradient types.
- Sandbox temp creation/cleanup failed for existing AppTest-from-string behavior,
  including with a workspace temp directory. Full regression reran with normal
  permissions and passed. No tests were skipped or changed to conceal this.
- Restricted uv cache access failed. The established cached `ty` tool succeeded
  with normal permissions.
- Browser inventory reported `The browser access host is unavailable`.
  Automatic approval review rejected selecting an unrelated existing Chrome
  window because reading it could expose private content outside the local-app
  validation task. That action was not bypassed. Browser acceptance is pending;
  explicit manual scenarios are in the acceptance document.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_seq2seq.py tests/test_seq2seq_ui.py tests/test_navigation.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
uv tool run ty check --python .venv learning/seq2seq_spec.py learning/seq2seq_adapter.py learning/seq2seq_training.py learning/seq2seq_ui.py learning/seq2seq_content.py visualization/seq2seq_graphs.py tests/test_seq2seq.py tests/test_seq2seq_ui.py learning/ui.py tests/test_navigation.py
.\.venv\Scripts\python.exe -m compileall -q learning/seq2seq_spec.py learning/seq2seq_adapter.py learning/seq2seq_training.py learning/seq2seq_content.py learning/seq2seq_ui.py visualization/seq2seq_graphs.py tests/test_seq2seq.py tests/test_seq2seq_ui.py
git diff --check
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8766 --server.enableAgentApi=true
```

## Delivery and remaining work

Feature branch: `feature/m11-transformer-encoder-decoder`; implementation commit
`92dcccd`, pushed. [Draft PR #9](https://github.com/swapnilpuri/deeplearningstudio/pull/9)
targets master. The GitHub connector's 403 was resolved by authenticated CLI PR
creation. Initial push auto-review rejection was resolved by verifying the exact
destination and explicit push instruction in the supplied request before retry.
Full delivery evidence is in [M11_HANDOFF.md](M11_HANDOFF.md). Master is not merged.
Detailed request, design, acceptance map, Volume 9 guide, roadmap and README are
included. M12 decoder scope is absorbed into the expanded M11 request; a new M12
scope requires architecture-owner definition. Suggested next action is browser
and independent educational acceptance, followed by that scope decision.
