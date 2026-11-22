# M11 handoff — Transformer encoder and decoder

- Status: Partial; implementation/automated validation delivered, browser and
  independent educational acceptance pending.
- Base: `41e8cbfa6dd7d6cc9681dadc1693c4f356add0d7`, fetched remote master with
  M10 PR #7 and shared horizontal navigation PR #8 merged.
- Branch: `feature/m11-transformer-encoder-decoder`.
- Implementation commit: `92dcccd`.
- Pushed branch; [draft PR #9](https://github.com/swapnilpuri/deeplearningstudio/pull/9)
  targets master. Nothing merged.

## Completed and decisions

All 18 functional areas are mapped in [acceptance](M11_ACCEPTANCE_CRITERIA.md).
Positions, encoder/decoder stacks, rectangular cross-attention, masks, full
model, toy copy/reverse tasks, teacher forcing, training, greedy inference,
visualization, tensor/parameter/complexity inspectors and fifteen graded checks.
Direct shared native tabs; lazy `tab.open` guards; `m11_` persisted widget keys;
existing progress v1; session-owned mutable models/results. No M1–M10 math
rewrites, new libraries, schema migrations or later milestone implementations.
See [requirements](M11_REQUIREMENTS.md), [design](M11_DESIGN.md),
[Volume 9](../VOLUME_9_ENCODER_DECODER.md) and [report](M11_COMPLETION_REPORT.md).

Batch-first shapes and explicit True=visible versus True=PAD conventions.
Both norm orders match PyTorch layers at float64 `1e-9` tolerance with mapped
weights and no dropout. Default CPU copy fit: loss `2.278946 → 0.015437`, greedy
training exact match `0/16 → 16/16` in 10.03s. No held-out generalization claim.

## Validation and gaps

- Full regression: **394 passed in 144.33s**, zero failures/skips.
- Final targeted regression after tensor-inspector clarification: **64 passed in
  31.26s**, zero failures/skips.
- Type check: all checks passed; compilation/diff checks passed.
- Live server: 23 ready M11 snapshots, no warning/error/exception elements;
  plus 3 ready snapshots for the final input/output-shape inspector check.
  Consoles clean; both temporary servers stopped.
- Initial sandbox runs had Windows temp/cache errors; normal-permission full
  regression and type checking succeeded. Report records all initial failures.
- Browser host unavailable. Automatic approval review rejected reading an
  unrelated Chrome window due to private-content risk; not bypassed.
- Visual/keyboard/narrow viewport/download dialogs, optional GPU, fresh install
  and independent educational review are unverified.

## Resume / next task

```powershell
git switch feature/m11-transformer-encoder-decoder
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

Choose Learning Studio → Transformer Encoder & Decoder. Execute the browser
scenarios in acceptance, review the educational content, and define the next
M12 scope. Do not merge into master without authorization.

## Git delivery

`git push -u deeplearningstudio feature/m11-transformer-encoder-decoder`
succeeded. Initial automatic approval review rejected publication because the
destination authorization was unclear. Read-only checks established that the
configured push URL exactly matched the user's supplied repository URL and that
the supplied request explicitly instructed push/PR creation. Retrying with that
evidence was approved; no destination or credentials changed.

The GitHub connector returned 403 `Resource not accessible by integration` for
PR creation. Authenticated `gh pr create --draft --body-file` succeeded at the
same repository. The temporary PR body and all test/live artifacts are ignored
under `.test-tmp-m11`; no dataset, cache, weights or secret was committed.
This documentation delivery commit follows `92dcccd` and records the actual PR.
