# Contributing to NeuroBranch

Thanks for your interest in NeuroBranch. It's an early, evolving MVP built and maintained as a
personal project, so the process here is intentionally lightweight — but a few conventions keep
things consistent.

---

## Before you start

For anything beyond a small fix (new cards, IPC changes, compiler/runtime behavior, agent tooling),
**open an issue first** to discuss the approach before writing code. This avoids wasted work on
changes that don't fit the project's direction — typed cards, two-way graph/PyTorch sync, and an
agent constrained to the same validated tools a user has.

Small, obvious fixes (typos, docs, a clear bug with a clear fix) can go straight to a PR.

---

## Getting set up

Follow [SETUP.md](SETUP.md) to install dependencies, provision the local Python runtime, and run
the app from source. Come back here once you have `npm run electron:start` working.

---

## Project shape

A quick map before you dig in — see [docs/project-overview.md](docs/project-overview.md) for more:

| Path | What lives there |
|---|---|
| `src/` | Renderer — React + React Flow canvas, PyTorch/split/player views, agent panel |
| `src/core/` | Framework-agnostic domain logic — typed IR, PyTorch compiler, layout, agentic graph editing |
| `electron/` | Electron main process — preload bridge, typed IPC, local Python runtime invocation, SQLite store |
| `scripts/` | Build/setup scripts, including `atomic_runtime.py` (the Python execution runtime) |
| `docs/` | Project overview and walkthroughs |

Core domain code (`src/core/`) should stay framework-agnostic — no React or Electron imports there.

---

## Making changes

1. Create a branch off `main`.
2. Keep changes focused — one logical change per PR. Don't mix refactors with feature work.
3. Match existing patterns in the file/module you're touching rather than introducing new ones.
4. If you're adding a new card type, IPC handler, or compiler rule, add or update the corresponding
   test alongside it.

---

## Before opening a PR

Run:

```bash
npm run lint          # oxlint — should report 0 warnings / 0 errors
npm run build          # tsc -b && vite build
npm run build:electron # tsc -p tsconfig.electron.json
npm test               # full vitest run
```

For changes touching the Electron main process, IPC, or the Python runtime, also run:

```bash
npm run test:desktop
```

A PR with failing lint, build, or tests won't be merged as-is — fix them first rather than opening
a WIP PR for maintainer triage.

---

## Commit messages

Keep them short and in the imperative mood, describing the *why* over the *what* where it isn't
obvious from the diff (e.g. `fix: prevent duplicate port ids on card duplicate`, not
`fix: update card.ts`). Conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`) are used
throughout the history — follow suit.

---

## Opening the PR

- Describe what changed and why, not just what — link the issue it addresses if one exists.
- Note any manual testing you did (especially for UI/canvas changes — a screenshot or short clip
  helps a lot).
- Keep the PR scoped to the discussed change; unrelated cleanups belong in a separate PR.

---

## Reporting bugs

Open an issue with:
- What you expected vs. what happened.
- Steps to reproduce (a minimal graph/card setup if relevant).
- Your OS and whether you're on the CPU or CUDA runtime (see [SETUP.md](SETUP.md#4-python-runtime-notes)).
- Relevant logs, if any (Electron main process logs, or `npm run electron:dev` console output).

---

## License

By contributing, you agree that your contributions will be licensed under the project's
[Apache-2.0 License](LICENSE).
