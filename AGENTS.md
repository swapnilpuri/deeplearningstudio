# Repository Agent Instructions — ChatGPT + Codex Workflow

For the full operating model see `docs/AI_DEVELOPMENT_OPERATING_MODEL.md`.

## Ownership and source of truth

- **ChatGPT web:** architecture, requirements, roadmap, educational/explanatory content, architectural review and milestone acceptance review.
- **Codex IDE/CLI:** repository inspection, implementation, automated tests, debugging, documentation maintenance, Git operations, and PR preparation.
- **Git + docs:** authoritative implementation, architecture decisions, test evidence, milestone status, and handoffs. Conversations are not the source of truth.
- **Human:** project priorities and consequential approvals; never bypass required approval gates.

## Project-specific settings

- Project name: DeepLearningStudio
- Stack and supported platforms: Windows 11, Python 3.12, PyTorch, Streamlit, Plotly, NumPy, scikit-learn, ONNX; uv
- Setup command: `uv venv --python 3.12`, then `uv pip install -r requirements.txt`
- Targeted test command: `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_rnn.py`
- Full test command: `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q`
- Run/smoke-test command: `.\.venv\Scripts\python.exe -m streamlit run app.py`
- Primary branch: `master`
- Requirements location: `docs/milestones/`
- Handoff location: `docs/milestones/<MILESTONE>_HANDOFF.md`

## Execution policy

1. On entry, inspect branch, `git status`, relevant requirement/handoff, and only task-related source files. Protect pre-existing uncommitted changes.
2. For nontrivial work, plan briefly and promptly perform the next safe, concrete action. Avoid prolonged speculation and architecture churn.
3. Implement one coherent, testable slice at a time; reuse established abstractions and preserve backward compatibility.
4. Run focused tests for each slice, and full regression before milestone acceptance or after broad shared changes.
5. Avoid redundant full-tree scans, repeated reads of unchanged files, and re-running expensive tests without a reason.
6. Use safe parallelism for independent reads/tests; do not parallelize edits or Git operations that share state.
7. Record exact test outcomes and gaps. Never claim GUI, platform, integration, or other tests passed unless executed.
8. Report compactly: changes made, files touched, commands/tests and results, risks, next task.
9. Update the milestone handoff at meaningful checkpoints. Keep details there, not in huge conversational recaps.
10. Use feature branches and PRs; do not merge to main/master automatically. Never overwrite user changes or force-push without approval.

## Speed tuning

- Prefer the smallest correct solution and focused scope rather than re-planning the full project.
- Start with Medium reasoning effort for typical implementation when available; lower for trivial tasks, higher for difficult architecture/debugging. Prompt text cannot control model inference latency or override product settings.
- A broad milestone specification is a reference: execute it in narrow, independently verifiable batches.
- No unnecessary ZIPs, duplicate project folders, speculative framework rewrites, or verbose repeated logs.
- Efficiency must not compromise correctness, tests, security, design consistency, or acceptance criteria.

## Learning-section navigation (M11 and future milestones)

All new DeepLearningStudio educational modules must use the standardized
horizontal learning-section navigation component. Dropdowns must not be used
as the primary mechanism for switching educational sections unless there is
an explicitly documented exception.

Use `learning.navigation.render_learning_section_navigation(sections, module_key)`.
Pass ordered labels or an ordered ID-to-label mapping and a unique module prefix.
Render each returned tab inside `with tab:` only when `tab.open` is true.
Use stable, module-prefixed input keys and `persist_state='session'` in lazy
panels. Keep training behind explicit buttons and retain results in session
state. Do not use button, upload or editor state as ordinary input state.
Native tabs supply styling and overflow; do not add custom CSS/JS or substitute
a dropdown for long section lists. Keep workspace/learning-path selectors separate.
The existing foundation, architecture and general training workspaces use
`lazy=False` to preserve their eager cross-panel dependencies and browser-only
tab switching. Dataset Management is a single workflow, so it needs no section tabs.
See `docs/milestones/UI_NAVIGATION_HANDOFF.md` for validation and manual review.

## Security and approval

Ask before destructive actions, discarding local changes, force pushes, schema/data migrations with irreversible impact, deployments, secret/configuration changes, or modifying permissions. Never store credentials in source, docs, commits, or logs.

## Required handoff format

Include milestone/status, base/current commit, branch, completed work, design decisions/links, tests (passed/failed/skipped/untested), risks, blockers, next task, and commands to resume. Use the latest code and test evidence rather than assuming the handoff is perfectly up to date.
