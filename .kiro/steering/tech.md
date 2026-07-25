# Tech

## Stack

- **Frontend**: React 19 + TypeScript, built with Vite 8.
- **Desktop shell**: Electron 43, packaged with electron-builder. Preload bridge
  (`electron/preload.cts`) enforces renderer isolation; IPC contract in
  `electron/ipc-contract.ts`.
- **Graph editor**: `@xyflow/react` (React Flow) with custom typed cables and layout.
- **Icons**: `lucide-react`.
- **Styling**: SCSS (`sass`), token-driven under `src/styles`.
- **Runtime**: Python 3 executes generated PyTorch graphs (`scripts/atomic_runtime.py`,
  deps in `requirements-runtime.txt`), bundled as an Electron extra resource.
- **AI**: `@openai/codex` App Server for ChatGPT sign-in; optional OpenAI API key fallback.
- **Installer**: Tauri-based bootstrap Setup in `apps/bootstrap-installer` (Rust + JS).

## Tooling

- **Lint**: `oxlint` (config `.oxlintrc.json`).
- **Test**: `vitest` with `jsdom`, `@testing-library/react`. Tests are colocated as
  `*.test.ts(x)` next to source.
- **TypeScript**: project references — `tsconfig.app.json` (renderer/core),
  `tsconfig.node.json`, `tsconfig.electron.json` (Electron main, output `dist-electron`).

## Common commands

```bash
npm install              # install deps
npm run dev              # web dev server (Vite)
npm run electron:dev     # runtime setup + build electron + Vite + Electron
npm run build            # tsc -b && vite build (web bundle)
npm run build:electron   # compile Electron main to dist-electron

npm test                 # full vitest run
npm run test:ci          # vitest excluding heavy runtime tests
npm run test:desktop     # runtime + build + desktop suite + smoke test
npm run lint             # oxlint

npm run electron:start   # build everything and launch Electron locally
npm run demo:agent       # guided agent demo
```

## Packaging

Unpacked dev builds: `package:mac:dir`, `package:win:dir`, `package:linux:dir`.
Signed/target builds: `package:mac`, `package:win`, `package:win:arm64`.
Artifacts land in `release/`. Version in `package.json` is the release source of truth;
`npm version <patch|minor|major>` runs `scripts/sync-release-version.mjs`.

## Notes for changes

- Never start long-running dev servers/watchers in automation; suggest the user run them.
- Use `getDiagnostics` to check TS/lint issues rather than ad-hoc shell commands.
- Keep renderer free of secrets; OpenAI keys stay encrypted via Electron `safeStorage`.
