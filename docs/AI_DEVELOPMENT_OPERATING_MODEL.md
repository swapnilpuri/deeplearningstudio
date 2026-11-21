# AI-Assisted Software Engineering — Operating Model

**Version:** 1.0
**Applicability:** Any software project, from small prototypes to multi-repository products
**Primary tools:** ChatGPT (web) for discovery, architecture, education, and independent review; Codex (IDE/CLI) for repository implementation and verification
**Source of truth:** Version-controlled code, tests, architecture decisions, and milestone records—not chat transcripts

> **Core principle:** Plan in ChatGPT. Implement and test in Codex. Preserve decisions, evidence, and handoffs in Git. Review against acceptance criteria before release.

## 1. Division of responsibilities

| Work | Default owner | Expected output |
|---|---|---|
| Problem framing, goals, user journeys, domain exploration | ChatGPT | Problem statement and scope |
| Architecture, design alternatives, trade-offs, domain boundaries | ChatGPT | Design/ADR/diagrams |
| Requirements, staged roadmap, acceptance criteria | ChatGPT | Versioned milestone specification |
| Practical implementation plan | ChatGPT + Codex | Tasks tied to existing repository structure |
| Source changes, refactoring, tests, dependency updates | Codex | Reviewed working-tree changes |
| Local execution, debugging, linting, test runs | Codex | Real command outputs and verified results |
| Git branches, commits, PR preparation | Codex | Commits and PR with traceability |
| Architectural consistency and milestone acceptance review | ChatGPT | Independent review and gap assessment |
| Product owner decisions and high-impact approvals | Human | Explicit approval recorded in repo where appropriate |

Roles are defaults, not tool restrictions. ChatGPT can inspect a repository when connected; Codex can propose design corrections. A tool's access and execution evidence determine what it can verify.

## 2. Project bootstrapping (once per new project)

1. Create a repository and a working directory; adopt a short project README with objectives and commands.
2. Place the companion `AGENTS.md` at the repository root. Adapt the sections marked *project-specific*.
3. Create `docs/ROADMAP.md`, `docs/architecture/`, `docs/adr/`, and `docs/milestones/` as warranted by project complexity.
4. In ChatGPT, develop scope, architecture, delivery milestones, assumptions, risks, test strategy, and measurable acceptance criteria.
5. Commit those specifications into the repository. Codex must read the **relevant** spec before coding.
6. Define minimum quality gates (unit/integration tests, lint/types, security, regression, documentation, and any platform-specific smoke tests).
7. Confirm Git remote and branch strategy; keep main/master protected by review when possible.

Avoid ceremony for tiny projects: a concise requirements file plus README and AGENTS.md may be enough.

## 3. Recommended per-milestone lifecycle

**A. Plan in ChatGPT** — Define user-visible outcomes, interfaces, architecture impact, constraints, demo scenarios, and acceptance criteria. Produce a concise versioned requirements file. Distinguish non-negotiable requirements from optional stretch goals.

**B. Baseline in Codex** — Check `pwd`, branch, `git status`, current commit, relevant files, and test commands. Never overwrite uncommitted user work. Run baseline tests when feasible and record results.

**C. Implement in focused batches** — Select the next independently verifiable slice. Inspect targeted files, make changes, run focused tests, correct failures, and update milestone status. Avoid full-repository scans and repeated architecture reconsideration without new evidence.

**D. Validate** — Run the full regression suite when the milestone or cross-cutting change is ready. Perform required smoke/integration/GUI/platform checks where executable. Explicitly record what was not tested.

**E. Review in ChatGPT** — Use committed code and `M*_HANDOFF.md`/completion reports for independent acceptance review. Address gaps in Codex. Do not treat conversational assurances as test evidence.

**F. Deliver** — Commit on a feature branch, open a PR to main/master, review and merge through the normal workflow. Do not require a ZIP when the project is already in Git and developed locally; package a release artifact only when deployment, distribution, or an explicit requirement needs one.

## 4. Codex efficiency policy

**Goal:** Reduce avoidable thinking, file scanning, and repeated execution, not essential reasoning or quality assurance.

- Start with a short action plan for complex tasks (roughly 3–7 steps); do not write lengthy speculative analyses before taking the first safe action.
- Read `AGENTS.md`, the relevant milestone requirement, and current handoff first; inspect only the files needed for the next task.
- Prefer established architecture and existing code over new frameworks or broad rewrites.
- Keep tasks bounded: one cohesive capability or bug cluster per batch. Use the full milestone specification as a reference, not as a request to redesign everything at once.
- Use targeted searches and inspections; do not repeatedly reopen unchanged files or re-run expensive commands without reason.
- Parallelize **independent read-only** inspections and tests when safe. Never race edits or Git writes to shared state.
- Run targeted tests after each batch; run full regression at appropriate checkpoints and before declaring a milestone complete.
- Avoid disabling tests, concealing failures, or reporting unexecuted checks as successful to save time.
- Present compact outcomes: changed files, tests, caveats, next task. Store detailed findings in the repository.
- For common implementation tasks, start with **Medium** reasoning effort when available; use lower effort for trivial mechanical edits and higher effort for difficult architecture/debugging. This is a tuning suggestion, not a speed guarantee or an instruction that overrides IDE settings.
- If the agent spends excessive time thinking without taking action, split the task further or start a new session from the handoff. A prompt cannot force faster model inference.

## 5. Git and safety

- The local repository is authoritative for uncommitted state; remote Git is authoritative for pushed history. Always inspect both as appropriate.
- Prefer feature branches and pull requests. Never push to protected main/master without the user/team's process.
- Ask before destructive actions: deleting user data, discarding uncommitted work, forced pushes, irreversible migrations, production deployments, or credential/security changes.
- Keep `.venv`, credentials, secrets, caches, generated datasets, and large unrelated binaries out of commits.
- Review staged diffs, respect `.gitignore`, and prevent accidental secret exposure.
- Reuse an existing workspace; don't make duplicate gigabyte-scale project copies merely as backups when Git versioning is sufficient.

## 6. Testing and evidence rules

Each completion report should separate **passed**, **failed**, **skipped**, **blocked**, and **not executed** checks. Record environment, exact commands, counts where available, significant failure traces, and commit SHA. A passing unit suite does not prove UI or integration behavior. Acceptance requires verification of the user's actual workflow, not just compilable code.

## 7. Session synchronization (ChatGPT ↔ Codex)

Do not attempt to merge chat histories. Instead, commit a durable handoff file:

`docs/milestones/<MILESTONE>_HANDOFF.md`

Keep it brief and current:

1. Project, milestone, status, branch, baseline/current commit SHA.
2. Scope and acceptance criteria (link to requirements).
3. Completed features and notable design decisions (link to ADRs).
4. Files changed and compatibility considerations.
5. Tests run with actual results; untested areas clearly marked.
6. Known defects, blockers, risks, and next highest-priority task.
7. Exact commands needed to continue.

Use the same file for fresh ChatGPT reviews and new Codex sessions; update at meaningful checkpoints rather than every tiny edit. Never treat it as stronger evidence than the current code and test output.

## 8. Prompt templates

### ChatGPT — start a new project or milestone

> Act as software architect and product/technical planning partner. Design [PROJECT / MILESTONE] for an implementation by Codex in the existing Git repository. Establish user outcomes, constraints, domain model, architecture and trade-offs, a staged roadmap, test strategy, risks, and measurable acceptance criteria. Output a concise versioned requirements/design package suitable for committing under `docs/`. Do not claim changes were implemented or verified. Follow `AI_DEVELOPMENT_OPERATING_MODEL.md`.

### Codex — implement the next batch

> Read root `AGENTS.md`, `docs/milestones/<MILESTONE>_REQUIREMENTS.md`, and the current handoff. Work in the existing repository; inspect `git status` and protect user changes. Choose the next highest-priority incomplete, independently testable requirement. Read only relevant files, implement the smallest correct change aligned with existing architecture, run focused tests, and update the handoff with real results. Avoid repeated repository scans and unnecessary redesigns. Continue with the next batch when safe. Run full regression at milestone completion; work on a feature branch; do not merge to main/master automatically.

### ChatGPT — review current progress

> Review the current [PROJECT] milestone from the latest accessible repository commit, requirements, ADRs, handoff, and completion report. Check each acceptance criterion against actual code and test evidence. Distinguish verified, incomplete, and unverified outcomes. Recommend the next focused Codex task and update architectural guidance as needed.

## 9. Definition of done

A milestone is complete only when its required functionality, tests, compatibility, documentation, and acceptance scenarios are verified, with any exceptions agreed and recorded. A handoff or PR is not proof of successful execution. If critical checks remain unverified, mark the milestone as incomplete or conditionally accepted—not fully complete.

## 10. Scaling across projects

Use the same operating model for AI learning platforms, business applications, microservices, orchestrators, and other projects. Keep the **generic** guideline stable; keep domain requirements, stack, commands, and project-specific security rules in each repository's `AGENTS.md` and milestone specs. For autonomous multi-agent projects, add explicit approval, authorization, environment, and recovery gates rather than letting “work autonomously” bypass them.
