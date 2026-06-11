# autocommit VS Code extension

This extension is a lightweight UI wrapper around the local `autocommit` CLI. It focuses on the primary review loop from issue #32: generate output quickly, let the user inspect and edit it, and only mutate Git after explicit approval.

## Commands

- `autocommit: Analyze Changes (Preview)` runs `autocommit analyze --json` and displays summaries, risk notes, dispatch status, and model/runtime errors in the side panel.
- `autocommit: Generate Commit Message (Preview)` generates an editable commit message from analysis output without mutating Git.
- `autocommit: Commit Approved Message` creates a Git commit from the edited message after confirmation.
- `autocommit: Preview Pull Request (Dry Run)` runs `autocommit pr --dry-run --no-interactive`.
- `autocommit: Create Pull Request` opens a terminal for the existing interactive `autocommit pr` flow.

Preview commands are intentionally labeled as previews or dry runs. Commands that create commits or pull requests are labeled as mutations and require explicit confirmation.

## Settings

- `autocommit.binaryPath`: CLI executable path. Defaults to `autocommit`.
- `autocommit.commitStagedOnly`: when true, the commit workflow only analyzes and commits staged changes.
- `autocommit.extraArgs`: extra CLI arguments for analysis/preview commands, such as model or profile overrides.

## Manual QA

1. Open a Git repository in VS Code with the `autocommit` CLI available on `PATH`.
2. Make a small source change.
3. Open the autocommit activity-bar view and run **Generate Commit Message (Preview)**.
4. Confirm that the side panel shows progress, generated output, an editable commit message, and no Git mutation has happened.
5. Edit the commit message and click **Commit Approved Message**.
6. Confirm the warning dialog distinguishes the Git mutation from preview actions, approve it, and verify that `git log -1 --pretty=%B` matches the edited message.
7. Run **Preview Pull Request (Dry Run)** and verify that no PR is created.
