# AGENTS.md

## Scope and priority

- Apply these instructions to the current directory and its subdirectories. More specific project conventions and deeper `AGENTS.md` files take precedence within their scope.
- Follow the user's current explicit instructions when they conflict with this file.

## Communication and workflow

- Communicate in English by default.
- Read relevant code, configuration, and errors before editing. For moderately complex or larger tasks, provide a brief plan with validation steps.
- For minor, reversible uncertainties, state assumptions and proceed. Ask before resolving ambiguity that materially affects architecture, data, dependencies, environments, CI/CD, or many files.
- Prefer the simplest solution consistent with the existing project. Keep plans and responses concise.

## Environment

- Development primarily uses Python on Linux. Prefer Linux-compatible commands and paths.
- Use GPU 1 (24 GB RTX 4090) by default. Check available disk space before storage-heavy jobs.
- Work within the current working directory; do not directly modify files outside it.
- Use the Conda environment `v2n` by default unless the user or project explicitly specifies another environment. Attempt execution in `v2n` first. If it fails, report the specific cause, such as missing packages, an unavailable environment, version incompatibility, or a code error.
- Use interpreter-bound commands such as `python -m pip` and `python -m pytest`. Never fall back silently to system Python or another environment.
- Do not install dependencies or modify the environment yourself. If packages are missing, identify them and ask the user to install them manually.

## Minimal changes

- Make only the smallest change needed to complete the task. Every changed line must serve the current objective.
- Follow existing style and structure, and reuse existing code. Avoid new abstractions, parallel implementations, or dependencies unless necessary.
- Do not perform unrelated refactoring, cleanup, renaming, formatting, or file moves. Mention unrelated issues without fixing them.
- Add tests, types, comments, or documentation only where they directly support the change.
- Avoid unnecessary wrappers, helper layers, and abstractions. Prefer direct, readable code unless encapsulation clearly reduces duplication or complexity.

## Data assumptions and error handling

- Assume input data is complete, valid, and conforms to the established contract unless the task or evidence indicates otherwise.
- Do not add speculative validation, repeated null checks, default values, fallback branches, or coercions for hypothetical malformed data.
- Keep the normal execution path straightforward. Avoid excessive or nested `try/except` blocks and broad catches such as `except Exception`.
- Catch only specific, expected exceptions when meaningful recovery or cleanup is required. Otherwise, let errors propagate.
- Never silently skip failed records, swallow exceptions, or return placeholder results to make execution appear successful.
- Preserve existing checks unless changing them is necessary for the task.
- For large-scale data generation, still account for interruption and resumption.

## Debugging and validation

- Establish the reproduction path and likely root cause using relevant evidence before fixing a bug. Prefer a minimal reproduction where practical.
- Fix the root cause. Distinguish code defects from dependency, interpreter, configuration, and environment problems.
- Run the smallest focused validation proportional to the change: reproduce and verify bug fixes, check the directly affected feature behavior, and preserve existing behavior during refactoring.
- Do not run full unittest discovery for a narrow or localized change. Reserve it for broad or high-risk changes, release-level validation, an explicit user request, or evidence that focused tests are insufficient.
- Avoid rerunning unchanged checks during the same task unless later edits could affect their result.
- Clearly distinguish validation that passed, failed, or was not run. Explain blockers and unconfirmed assumptions; never claim unexecuted checks passed.

## Safety

- Obtain explicit authorization before destructive operations, overwriting existing work, rewriting Git history, force-pushing, or modifying production, credentials, permissions, CI/CD, database migrations, or global environment settings.
- Before such operations, explain the action, reason, scope, and safer alternatives.
- Do not run destructive commands such as `rm -rf`, `git reset --hard`, `git clean -fd`, or `git push --force`, or overwrite configuration, lockfiles, or migrations without explicit confirmation.
- Prefer read-only inspection, local edits, and reviewable diffs.