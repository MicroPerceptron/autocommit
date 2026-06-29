# autocommit VS Code extension

Review and approve local [autocommit](https://github.com/MicroPerceptron/autocommit) analysis, commit, and PR workflows from VS Code.

## Commands

| Command | Description |
|---------|-------------|
| `autocommit: Analyze Changes (Preview)` | Run `autocommit analyze --json`, display structured report |
| `autocommit: Generate Commit Message (Preview)` | Run `autocommit commit --dry-run --json`, editable message |
| `autocommit: Commit Approved Message` | Stage + commit using the approved message (via `autocommit commit -m`) |
| `autocommit: Preview Pull Request (Dry Run)` | Run `autocommit pr --dry-run --no-interactive` |
| `autocommit: Create Pull Request` | Run `autocommit pr --interactive` in a terminal with configured flags |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `autocommit.binaryPath` | `"autocommit"` | Path to the CLI binary |
| `autocommit.commitStagedOnly` | `false` | Stage-only mode |
| `autocommit.extraArgs` | `[]` | Extra CLI args (e.g. `--profile`, `--model-path`) |
| `autocommit.prDraft` | `false` | Always create PR as draft |
| `autocommit.prBase` | `""` | Default base branch |
| `autocommit.prReviewers` | `[]` | Default PR reviewers |
| `autocommit.prLabels` | `[]` | Default PR labels |

## Workflow

1. **Analyze** or **Generate Commit Message** (preview only, no Git mutation)
2. Inspect the output and edit the commit message if needed
3. **Commit Approved Message** — stages changes and commits via `autocommit commit -m`
4. **PR Preview** — dry-run pull request creation
5. **Create PR** — opens a terminal with `autocommit pr --interactive` for issue linking

## Manual QA

- Verify the webview panel appears in the activity bar ("autocommit" icon)
- Run Analyze/Generate on a repo with changes — output should appear
- Edit the commit message, then commit — should create a real Git commit
- Run PR Preview — should show dry-run output
- Run Create PR — should open a terminal
