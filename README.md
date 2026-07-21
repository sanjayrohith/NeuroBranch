# NeuroBranch

NeuroBranch is a desktop laboratory for composing neural architectures from executable atomic blocks. It combines a visual graph editor, generated PyTorch, step-by-step execution, and a constrained AI graph planner.

- [Product page and Setup downloads](https://www.complexity-ai.fr/labo-ai)
- [Narrated demo](https://youtu.be/Fv_PP0xiTj0)
- [OpenAI Build Week submission](https://devpost.com/software/labo-ai)

## OpenAI Build Week: Codex and GPT-5.6

NeuroBranch was created during OpenAI Build Week with Codex as the primary engineering environment and GPT-5.6 as the reasoning model used for the core build. The repository's dated commit history records the implementation during the submission period.

Codex accelerated the project across the Electron, React, TypeScript and Python boundaries. It was used to inspect and refactor the evolving graph editor, implement typed card contracts and graph/PyTorch synchronization, diagnose the packaged Python runtime, build SQLite workspace persistence, harden encrypted API-key handling, create macOS and Windows packages, and repeatedly run the desktop validation suite. Codex also drove the automated demo workflow and helped identify cross-architecture failures that were difficult to reproduce manually.

GPT-5.6 was used inside Codex for the cross-cutting implementation and debugging work. On desktop, Ask LABO can use the judge's ChatGPT session through the official Codex App Server; an explicitly configured OpenAI API key remains an optional fallback (`gpt-5.6-terra`). Ask LABO receives a bounded view of the card catalog and current graph, then plans graph changes that are validated locally before they can mutate the graph.

The key product decisions remained human-directed: use atomic typed cards instead of an unrestricted code editor; preserve existing architectures as read-only in parallel mode; separate review from auto-apply; make architecture deletion explicitly targeted; give reusable-card creation its own full-canvas visual workspace; encrypt user API keys locally; and represent parallel computation through stable topology-aware XY placement. Codex helped turn those decisions into tested product behavior and iterate quickly when the first versions were ambiguous or unsafe.

## Current capabilities

- More than 100 typed, executable model cards grouped by useful families.
- Blank, GPT-like, learned-MoE, token-routing, and TR 300M starters.
- Executable Vision Transformer, multimodal image-editing, spatiotemporal video, and audio encoder starters with dedicated media atomics and tokenizer pipelines.
- Drag-and-drop graph composition with typed elastic cables and deterministic topology-aware XY placement that preserves parallel lanes and minimizes crossings.
- Two-way graph/PyTorch synchronization for supported semantic atoms.
- Dedicated full-canvas reusable-card studio with Blocks, PyTorch and Split views, category-aware composition, typed plugs, agent assistance and a safe generated `nn.Module` preview.
- Separate add and edit modes; existing cards open in a central editor and user-created library cards can be deleted.
- Natural-language card search across native atomics and graph inputs.
- Contextual card construction palettes whose operations and typed plugs follow the selected category.
- Vector SVG diagram and generated PyTorch export through the desktop save dialog.
- Atomic PyTorch player with run, rerun, reset, and step-by-step execution.
- Tokenizer and training studios.
- Ask LABO planner that can add available blocks, connect compatible ports, or report missing capabilities.
- Parallel-aware automatic graph layout and multi-architecture composition without overwriting the current canvas.
- SQLite-backed graph drafts, cards, optimizer configurations and user presets, preserved across application updates.
- Desktop **Continue with ChatGPT** sign-in through the official Codex App Server, plus optional per-user OpenAI API-key fallback through Electron encrypted storage.

## Judge quick start — one command

The commands below always install the latest published release. Each bootstrap script downloads the small native Setup helper, verifies its published SHA-256 digest, and launches the source-first installer.

### macOS (Apple silicon)

Paste this single command into Terminal:

```bash
curl -fsSL https://github.com/Complexity-ML/labo-ai/releases/latest/download/install-labo-ai-macos.sh | bash
```

### Windows (x64)

Paste this single command into PowerShell:

```powershell
irm https://github.com/Complexity-ML/labo-ai/releases/latest/download/install-labo-ai-windows.ps1 | iex
```

### Linux (x64)

Paste this single command into a terminal:

```bash
curl -fsSL https://github.com/Complexity-ML/labo-ai/releases/latest/download/install-labo-ai-linux.sh | bash
```

**NeuroBranch Setup** then fetches the latest tagged source, verifies and provisions its own Node.js runtime, and builds the Electron application locally without replacing private workspace data. Graph editing and the player need no account. Ask LABO can use a ChatGPT account on desktop or an optional user-provided API key.

Supported packages:

- macOS 12 or later on Apple silicon: `LABO-AI-Setup-arm64.dmg`.
- Windows 10/11 x64: `LABO-AI-Setup-x64.exe`.
- Linux x64: `LABO-AI-Setup-x64.AppImage` (generic AppImage, installed for the current user).

The Setup packages are currently unsigned, so macOS Gatekeeper or Windows SmartScreen may request confirmation on first launch. The one-command path runs the checksum-verified helper directly; the DMG, EXE and AppImage remain available from [GitHub Releases](https://github.com/Complexity-ML/labo-ai/releases/latest) for manual installation. The Electron application is produced locally rather than downloaded as an opaque prebuilt binary. Internet access and several minutes are required for the first install; later updates reuse the managed Node.js runtime. Setup also updates itself from verified release assets before rebuilding NeuroBranch when a newer helper is available.

Suggested test path:

1. Open **Blank starter** and drag typed cards from the block library.
2. Open **Ask LABO**, choose **New parallel** and **Auto apply**, then request a compact GPT-like QA architecture.
3. In **Settings → Agent**, choose **Continue with ChatGPT**. Alternatively, add an OpenAI API key; it is encrypted locally with Electron `safeStorage` and is never returned to the renderer.
4. Run the resulting graph with the atomic player, then inspect the **PyTorch** and **Split** views.
5. Save the workspace as a preset and export the graph as SVG or the generated model as Python.

Graph editing, built-in presets, PyTorch inspection, local execution and export can be tested without an account or API key. Ask LABO requires either the desktop ChatGPT connection or an OpenAI API key. The web build remains API-key/account based and does not reuse a browser ChatGPT session.

## Development

Requirements:

- Node.js and npm
- Python 3 with the packages in `requirements-runtime.txt`
- macOS, Windows or Linux for the packaged desktop build

```bash
npm install
npm run electron:dev
```

Run the validation suite:

```bash
npm test
npm run lint
npm run test:desktop
```

## Desktop build

Build and start Electron locally:

```bash
npm run electron:start
```

Create an unpacked macOS Electron application for development:

```bash
npm run package:mac:dir
```

Create an unpacked Windows Electron application for development:

```bash
npm run package:win:dir
```

Create an unpacked Linux Electron application for development:

```bash
npm run package:linux:dir -- --x64
```

Build the public source-first Tauri Setup package:

```bash
npm ci --prefix apps/bootstrap-installer
npm run build:mac --prefix apps/bootstrap-installer # macOS
npm run build:win --prefix apps/bootstrap-installer # Windows
npm run build:linux --prefix apps/bootstrap-installer # Linux AppImage
```

Run the guided 2–3 minute agent demo after connecting ChatGPT or adding an API key in the app:

```bash
npm run demo:agent
```

The sequence starts with a conversational `Hello`, builds a compact GPT-like QA graph, then asks the agent to upgrade it into a token-routed residual MoE before running it and showing synchronized PyTorch. Use `npm run demo:agent -- --keep-open` while recording, or add `--no-cues` when recording a clean voice-over-only version.

The recording sequence and hackathon copy are in [`docs/hackathon-submission.md`](docs/hackathon-submission.md).

Artifacts are written to `release/`. The Python atomic runtime is bundled as an Electron extra resource so packaged builds do not resolve it through the application archive.

### Publish a desktop release

The package version is the release source of truth. After the release changes are committed on `main`, bump it and push the generated commit and tag:

```bash
npm version patch
git push origin main --follow-tags
```

The `Desktop release` workflow verifies that the `vX.Y.Z` tag matches both the application and Setup versions, validates the project, builds the small Apple-silicon DMG, Windows x64 Setup EXE and Linux x64 Setup AppImage, and publishes them with stable filenames. The Setup then builds Electron locally from that latest tagged release. Existing installations expose the same updater from the shared **Settings → General** page. The Complexity website can use GitHub's `/releases/latest/download/…` URLs, so its download buttons do not require a separate version edit.

### Source-first installation layout

- Setup state and its managed Node.js runtime live in the operating system's local application-data directory.
- macOS installs NeuroBranch to `~/Applications/NeuroBranch.app`; Windows installs it below `%LOCALAPPDATA%/Programs/NeuroBranch`; Linux installs it below `${XDG_DATA_HOME:-~/.local/share}/NeuroBranch/app` and creates a per-user `.desktop` launcher.
- One previous application directory is retained for rollback.
- Before each application update, the installed Setup helper checks and SHA-256 verifies its own latest native helper, relaunching it when the updater itself changed.
- The Setup window exposes live stages, progress and an expandable installation journal, then quits automatically when NeuroBranch launches.
- SQLite workspaces, custom cards and optimizer presets remain in Electron's separate user profile and are never replaced by an application update.
- The updater accepts no repository or command from the renderer: its source repository and build commands are fixed in the native Setup code.

## Security model

- Renderer isolation is enforced through the Electron preload bridge.
- ChatGPT OAuth and token refresh are owned by the bundled Codex App Server; session tokens are never sent through the renderer.
- OpenAI keys are encrypted with Electron `safeStorage` and never exposed to the renderer after saving.
- Ask LABO receives a bounded graph context and must return a strict structured plan.
- Custom PyTorch cards accept only explicitly supported `torch.nn` constructors and literal arguments.
- Generated graphs run through the dedicated local Python runtime, not through arbitrary shell evaluation.
