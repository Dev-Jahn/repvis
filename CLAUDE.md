# repvis — Claude Code project instructions

<!-- jahns-workflow:begin (managed block — edit via /jahns-workflow:init) -->
## Workflow (jahns-workflow)

- **SSOT**: none — repvis has no single design doc; `PROGRESS.md` + `README.md` + code docstrings + commit messages are the design record. SSOT features are disabled in `.jahns-workflow.yml` (no §-anchor citing, no ADR-gated SSOT edits, no section digest). The ADR series still records repo-wide decisions.
- **Task registry**: every unit of work gets an ID `<type>/<kebab-slug>` (feat|fix|perf|gate|spike|decision|docs|chore) registered in `tasks.yaml` with an explanatory title BEFORE first use. Bare codenames (P0, E3, Q1…) are banned. `ROADMAP.md` is generated — never edit it.
- **Read & mutate the registry through the CLI, not raw** (it grows to thousands of lines): `jw task list [--status/--type/--milestone/--round]` and `jw task show <id>` to read; `jw task add <id> --title … [--severity/--deps/…]`, `jw task set <id> <field> <value>`, `jw task drop <id>` to mutate (validated, comment-preserving). Reading `tasks.yaml` whole is redirected here by a hook.
- **Severities** on review findings: blocker > major > minor (field, not ID). Blockers resolve before the next round.
- **Rounds**: close each work round with `/jahns-workflow:round` (updates registry, PROGRESS, roadmap, review packet). Ingest external review replies with `/jahns-workflow:review`.
- Full convention: `docs/CONVENTIONS.md`.
<!-- jahns-workflow:end -->
