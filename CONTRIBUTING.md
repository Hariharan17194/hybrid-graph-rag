# Contributing

Thanks for your interest! This project follows a small set of conventions so history stays readable and releases stay automatic.

## Setup

```bash
git clone https://github.com/Hariharan17194/<repo>.git && cd <repo>
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

## Branch naming

`<type>/<short-kebab-description>` — optionally with an issue number.

| Branch | Use for |
|---|---|
| `feat/bm25-retriever` | New feature |
| `fix/42-empty-pdf-crash` | Bug fix (issue #42) |
| `docs/readme-architecture` | Documentation only |
| `refactor/split-pipeline` | Code change, no behaviour change |
| `chore/bump-deps` · `ci/cache-pip` | Tooling, deps, CI |

`main` is always deployable. Never commit directly to it — open a PR, even for solo work (it's what CI and release automation hook into).

## Commit messages — Conventional Commits

```
<type>(<optional scope>): <imperative summary, ≤ 72 chars>

<optional body: what and why, not how>

<optional footer: Closes #12 · BREAKING CHANGE: …>
```

| Type | Version bump | Example |
|---|---|---|
| `feat` | minor (0.**x**.0) | `feat(rag): add BM25 keyword retriever` |
| `fix` | patch (0.0.**x**) | `fix(upload): reject empty PDFs with 422` |
| `feat!` / `BREAKING CHANGE:` | major | `feat(api)!: rename /query to /ask` |
| `docs` `style` `refactor` `perf` `test` `build` `ci` `chore` | none | `docs: add architecture diagram` |

A `commit-msg` hook (installed above) rejects non-conforming messages.

## Pull requests

1. Keep PRs small and focused — one concern per PR.
2. The **PR title** must be a Conventional Commit (it becomes the squash-merge commit).
3. CI must be green: ruff lint + format, pytest, secret scan, Docker build.
4. Squash-merge. release-please turns merged commits into a CHANGELOG and a tagged release.

## Code style

- Formatting and linting: **ruff** (`ruff check . --fix && ruff format .`)
- Type hints on public functions
- Tests live in `tests/` and must not call paid APIs — mock the LLM client.
