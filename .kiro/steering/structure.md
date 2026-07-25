# Structure

## Top level

- `src/` — React renderer app (UI, core logic, studios).
- `electron/` — Electron main process, preload bridge, IPC contract, desktop services.
- `scripts/` — build/runtime helpers, agent demos, installers, Python runtime.
- `apps/bootstrap-installer/` — Tauri source-first Setup helper (Rust + JS/SCSS UI).
- `public/` — static assets served by Vite.
- `docs/` — project overview and reference docs (overview, demo walkthrough).
- `build/`, `release/` — icons/build inputs and packaged artifacts (generated).

## `src/` layout

- `src/App.tsx` — root component; `src/main.tsx` — entry.
- `src/core/` — framework-agnostic domain logic: graph IR, cards/atoms, execution
  plans, PyTorch compiler/dialect, tokenizer, stats, presets, layout/placement.
  This is the heart of the model; keep it pure and well-tested.
- `src/model/` — model studio UI (graph canvas, export, agent plan review, card creator).
- `src/studio/` — shared studio shell, panels, header, search, settings UI.
- `src/tokenizer/`, `src/training/` — tokenizer and training studio features.
- `src/styles/` — SCSS partials (`_name.scss`) and token definitions.
- `src/test/` — shared test setup.
- `src/electron-api.d.ts`, `src/browser-api.ts` — platform bridge typings/adapters.

## Conventions

- Tests are colocated with source as `*.test.ts` / `*.test.tsx`.
- Domain/pure logic lives in `src/core`; UI components import from it, not vice versa.
- Keep the renderer isolated from the Electron main process; cross the boundary only
  through the preload bridge and typed IPC contract (`electron/ipc-contract.ts`).
- SCSS partials are prefixed with `_` and imported into aggregate stylesheets; use
  design tokens rather than hard-coded values.
- Prefer small, typed modules; card/atom definitions are typed contracts.

## Platform boundaries

- Web build and desktop build share `src/`; guard desktop-only behavior behind the
  platform APIs rather than assuming Electron is present.
- The Python runtime (`scripts/atomic_runtime.py`) is invoked from the main process,
  never from the renderer directly.
