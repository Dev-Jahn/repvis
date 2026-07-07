# ADR-0000: Adopt the jahns-workflow harness for repvis

- Status: accepted
- Date: 2026-07-07
- Round: bootstrap (pre-round adoption)
- SSOT sections affected: none (SSOT features disabled for this project)
- Tasks: chore/adopt-jahns-workflow

## Context

repvis grew from a first commit (2026-06-23) into a decision-dense tool: a fully-GPU
decode→extract→encode pipeline, per-model dtype/compile handling, and a subtle remove_bg
method — with the rationale and dead ends scattered across README, commit messages, and code
docstrings, and no task registry to track open work. It is a solo-developer-plus-agents repo
(public at github.com/Dev-Jahn/repvis) that already works round-based with direct commits to a
linear `main`. Adopting jahns-workflow gives it a shared task-ID grammar, rounds, ADRs, and a
review cycle, consistent with the developer's other projects.

## Decision

Adopt jahns-workflow v1 conventions (docs/CONVENTIONS.md) from this commit onward — not applied
retroactively to existing docs, commits, or history. Configuration:

- **review.mode: packet** (reviewers: codex, gpt-5.5-pro; `require_ci: false`) — a round closes,
  is pushed, and hands a web reviewer the round's request markdown; fits the existing
  direct-to-`main`, linear-history flow.
- **SSOT disabled** — repvis has no single canonical design/spec document; PROGRESS.md + README +
  code docstrings + commit messages are the design record. `ssot:` is omitted from the config,
  so §-anchors, ADR-gated SSOT edits, and section digests are inactive (the ADR series itself
  stays active as the decision log).
- Task IDs `<type>/<slug>` registered in `tasks.yaml` before first use; rounds `YYYY-MM-DD-<slug>`;
  ADRs one monotonic numeric sequence; PROGRESS.md as the work log.

## Consequences

- Easier: cross-project consistency (same ID grammar as the developer's other repos), traceable
  open work, ADR-gated decisions, a repeatable round/review cycle.
- Harder: a small per-round bookkeeping overhead (close rounds via `/jahns-workflow:round`).
- Follow-up: the open loose ends from PROGRESS.md were seeded as `tasks.yaml` entries.
- Invalidated if repvis later grows a canonical design/spec doc worth binding — then enable SSOT
  (create DESIGN.md, add `ssot:` to the config) via a new ADR.

## Alternatives considered

- No harness (ad-hoc PROGRESS + git) — rejected: loses task traceability and cross-project grammar.
- Enable SSOT now (DESIGN.md skeleton) — rejected: repvis has no canonical spec; the SSOT
  machinery's overhead is unjustified for a tool whose truth is its code. Revisit per above.
- pr review mode — rejected: repvis commits directly to a linear `main`; PR-per-round + a merge
  gate is heavier than the current flow needs.
