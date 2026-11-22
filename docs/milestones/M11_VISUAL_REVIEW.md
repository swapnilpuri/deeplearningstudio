# M11 Edge visual review

Date: 2026-11-22 (America/Chicago). Follow-up to implementation commit `92dcccd`
and delivery documentation `157ba1e` on `feature/m11-transformer-encoder-decoder`.

The user explicitly authorized a dedicated local-app browser window. Native
Windows Computer Use accessed the user-opened Microsoft Edge window and verified
its address bar/document URL as `http://127.0.0.1:8766/`. The browser-extension
host remained unavailable, but native browser interaction succeeded. Earlier
URL-detection and unrelated-window blockers were not bypassed.

## Observed checks

| Area | Executed and observed |
| --- | --- |
| Desktop layout | Dark theme, existing 80% browser zoom, approximately 1676×994 window. All seven M11 tabs visited; active underline, spacing, lazy panel changes and text/tables displayed. |
| Existing studios | Foundation workspace and M10 lesson opened. M10's native tabs have the same active underline, horizontal layout and overflow convention as M11. This supplements, rather than replaces, automated M1–M10 regression. |
| Keyboard | Right arrow + Enter moved from Encoder to Decoder. After reducing the window to approximately 500 pixels wide, the overflowing final Exercises & Progress tab remained reachable by keyboard and its content rendered. |
| Narrow layout | Sidebar collapsed automatically; headings and exercise text wrapped. Tabs remained horizontal with overflow. Window size was restored after checking. |
| Positions | Token, position and combined tensor tables rendered. Default sinusoidal position 0 alternated 0/1 across D=16. Position heatmap rendered; corrected axes verified as Embedding dimension / Position index. |
| Encoder/decoder | Layer selectors, stage explanations, tensor samples and shapes rendered. Decoder self-attention heatmap displayed forbidden future cells with zero probability. |
| Masks | Encoder source-PAD visibility displayed a forbidden final key column. Switching to decoder causal+target padding displayed the lower-triangular visibility matrix and negative-infinity forbidden logits. |
| Cross-attention | Rectangular three-query/four-key heatmap rendered with source labels on key columns, decoder labels on query rows, a 0–1 scale and a zero-probability PAD column. Q/K tables and head/layer selectors displayed. |
| Training | Explicitly ran the default 150-step copy experiment. Spinner appeared only after Train. Loss, teacher-forced token/sequence accuracy and pre-clip gradient curves rendered; greedy exact match displayed 0.0 → 1.0 and before/after prediction rows rendered. |
| Training retention | After visiting Quiz and Exercises, Concepts offered the matching trained model and examples. Navigation did not visibly restart training. Exact training-call-count retention is also covered by AppTest. |
| Quiz draft | Selected the first question's answer, left for Exercises, and returned; selection remained. Full fifteen-question grading/feedback was already exercised by AppTest and the earlier live agent-API run; it was not manually repeated for every question here. |
| Notes/completion | Entered and saved a test note, marked seq2seq complete, and observed both values in the progress JSON. Notes survived section changes. |
| Export | Export opened Edge's Downloads UI with newly generated `learning-progress (4).json`. Existing downloads were not overwritten or opened. |
| Import | Used the Windows file picker to select that generated file and clicked Load progress. Repeated after a clean server/browser session with visibly empty progress: seq2seq completion and the saved note were restored. The notes-editor issue described below was fixed and verified in Edge. |
| Architecture | Corrected vertical diagram rendered readable encoder/decoder paths, residual edges, LayerNorm/FFN ordering, encoder memory K/V connection and vocabulary projection. Tensor-stage selector and numerical input/output shape inspection rendered. |

Screenshots were inspected through the Computer Use tool. No private browser
window contents, unrelated files, authentication settings or OS preferences were
used. The exported test JSON contains only the synthetic completion/note record.

## Issues found and corrected

1. **Position heatmap axes:** the reused attention plot labeled positional
   dimensions as tokens. Added explicit axis overrides for this explorer.
2. **Unreadable architecture diagram:** the original long horizontal graph
   compressed labels severely. Changed to a vertical layout with compact multiline
   nodes and readable source/target paths. An intermediate cross-cluster
   `rank=same` constraint prevented frontend rendering; removed it and verified
   the final diagram in Edge. Layer equations and model computation are unchanged.
3. **Imported notes editor:** fresh-session import restored saved notes into
   progress but left the existing editor draft blank. Import now queues the saved
   note and applies it before creating the editor on the next rerun. Confirmed
   that the editor displays imported notes. Added an AppTest covering draft
   replacement, navigation, safe subsequent save and preservation of earlier
   lesson notes/scores/completion.

Focused UI results: initial two-presentation-fix suite **5 passed in 50.23s**;
after import regression test, **6 passed in 37.74s**. Final type, compilation and
diff checks passed. Final full regression is recorded in the completion report.

A clean server restart was used during diagram verification; loss of in-memory
training state across that restart is expected. The previously exported progress
file successfully restored notes/completion into the new session. During one
forced browser/server reload, the Windows asyncio transport logged a
`ConnectionResetError [WinError 10054]`; no corresponding application warning or
exception panel appeared. Do not describe that console as entirely clean.

## Remaining review boundaries

This verifies core browser workflows at the observed desktop/narrow sizes and
current Edge version, not every possible configuration, browser or assistive
technology. Learned positions, both norm orders and additional layer counts have
automated and live agent-API coverage; this browser pass did not repeat all such
combinations. Optional GPU, fresh dependency installation and independent
educational acceptance remain unexecuted. No merge or final owner acceptance is
implied by this visual review.
