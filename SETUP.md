# Local Setup & Run

How to set up and run **NeuroBranch** locally from source. This is the developer path
(build + launch the Electron app yourself). For the one-command end-user installer, see the
[README](README.md#quick-start--one-command).

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| **Node.js** | 20+ (tested on 22) | includes `npm` |
| **Python** | 3.10+ (tested on 3.14) | used for the local model runtime |
| **OS** | Linux, macOS, or Windows | Electron desktop app |
| **Display** | required to launch the app | Linux: an X11/Wayland session (`DISPLAY` / `WAYLAND_DISPLAY` set) |
| **Free disk** | ~2 GB (CPU runtime) / ~8 GB+ (CUDA runtime) | see [Python runtime notes](#4-python-runtime-notes) |

Check your toolchain:

```bash
node -v && npm -v && python3 --version
```

---

## 2. Quick setup (happy path)

From the project root:

```bash
npm install            # JS/TS dependencies (~1 GB in node_modules)
npm run runtime:setup  # creates .venv and installs torch + tokenizers
npm run electron:start # builds web + electron, then launches the app
```

`electron:start` runs `runtime:setup`, builds the web bundle and the Electron main, then opens
the desktop window. If `runtime:setup` already reports "ready", it is skipped automatically.

> **Low on disk?** Read [section 4](#4-python-runtime-notes) *before* running `runtime:setup` —
> the default install pulls the multi-GB CUDA build of PyTorch. Use the CPU-only path if space
> is tight or you don't have an NVIDIA GPU.

---

## 3. What each step does

```bash
npm install               # install locked JS deps
npm run runtime:setup     # provision .venv (Python) with runtime-requirements.txt
npm run build             # tsc -b && vite build  -> dist/        (web bundle)
npm run build:electron    # tsc -p tsconfig.electron.json -> dist-electron/ (main/preload)
npm run electron:start    # runtime:setup + build + build:electron + launch Electron
npm run electron:dev      # runtime:setup + build:electron + Vite dev server + Electron (hot reload)
```

- The renderer loads `dist/index.html` in production (`electron:start`).
- In dev (`electron:dev`), it loads the Vite dev server via `VITE_DEV_SERVER_URL`.
- The Python runtime (`scripts/atomic_runtime.py`) is invoked by the Electron main process to
  execute graphs; it is not required for graph editing, PyTorch view, or export.

---

## 4. Python runtime notes

`requirements-runtime.txt` pins `torch==2.13.0`. On Linux, the default PyPI index resolves this to
the **CUDA** build, which downloads several GB of `nvidia-*` packages.

### Option A — CPU-only (recommended for laptops / low disk / no GPU)

Much smaller (~few hundred MB). Run this **instead of** `npm run runtime:setup`:

```bash
.venv/bin/python -m pip install --no-cache-dir --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements-runtime.txt

# tell runtime:setup the venv is ready so it won't reinstall the CUDA build:
sha256sum requirements-runtime.txt | awk '{print $1}' > .venv/.neurobranch-runtime-requirements.sha256
```

If `.venv` doesn't exist yet, create it first: `python3 -m venv .venv`.

### Option B — default (CUDA) build

Only if you have an NVIDIA GPU and enough disk:

```bash
npm run runtime:setup
```

### How readiness is tracked

`runtime:setup` writes `.venv/.neurobranch-runtime-requirements.sha256` (a hash of
`requirements-runtime.txt`). On later runs it skips reinstalling if the hash matches **and** the
imports (`torch`, `tokenizers`, ...) succeed. To force a clean reinstall (e.g. to switch CPU↔CUDA):

```bash
rm .venv/.neurobranch-runtime-requirements.sha256
# then run Option A or B again
```

You can point setup at a specific interpreter with `NEUROBRANCH_AI_PYTHON=/path/to/python`.

---

## 5. Running the app

```bash
# Production-style: build everything and launch
npm run electron:start

# Launch only (if already built)
node_modules/.bin/electron .

# Dev mode with hot reload (long-running; run in its own terminal)
npm run electron:dev
```

Launch it **detached** (keeps running after the terminal closes), Linux example:

```bash
DISPLAY=:0 setsid nohup node_modules/.bin/electron . >/tmp/neurobranch-app.log 2>&1 &
```

Stop the app:

```bash
pkill -f 'electron/dist/electron .'
```

---

## 6. Validate the setup

```bash
npm run lint        # oxlint — should report 0 warnings / 0 errors
npm test            # full vitest run
npm run test:ci     # vitest excluding heavy runtime tests (faster)
npm run test:desktop # runtime + build + desktop suite + smoke test
```

A green `npm run build` + `npm run build:electron` + `npm run lint` is a solid readiness signal.

---

## 7. Troubleshooting

**`ERROR: No space left on device` during `runtime:setup`**
The CUDA torch build filled the disk. Reclaim space and use the CPU build:
```bash
.venv/bin/python -m pip cache purge     # frees cached wheels (regenerable)
du -sh ~/.cache/*                       # find other reclaimable caches
df -h /                                 # check free space
```
Then follow [Option A](#option-a--cpu-only-recommended-for-laptops--low-disk--no-gpu).

**`UserWarning: Failed to initialize NumPy: No module named 'numpy'`**
Harmless — torch works without it. If a step-through execution ever errors on numpy, add
`numpy` to `requirements-runtime.txt` and rerun the runtime install.

**`'--ozone-platform=wayland' is not compatible with Vulkan` / Fontconfig cache warning (Linux)**
Benign. If rendering misbehaves on Wayland, force X11:
```bash
node_modules/.bin/electron . --ozone-platform=x11
```

**App doesn't open / no window**
Ensure a display is available (`echo $DISPLAY` / `echo $WAYLAND_DISPLAY`) and check the log
(`/tmp/neurobranch-app.log` if launched as above).

**`runtime:setup` says Python 3.10+ required**
Set a compatible interpreter: `NEUROBRANCH_AI_PYTHON=$(which python3.11) npm run runtime:setup`.

---

## 8. Command reference

| Command | Purpose |
|---|---|
| `npm install` | Install JS/TS dependencies |
| `npm run runtime:setup` | Provision the Python `.venv` (CUDA by default) |
| `npm run build` | Build the web bundle → `dist/` |
| `npm run build:electron` | Build the Electron main → `dist-electron/` |
| `npm run electron:start` | Build everything and launch the app |
| `npm run electron:dev` | Dev server + Electron with hot reload |
| `npm run lint` | Lint with oxlint |
| `npm test` / `npm run test:ci` | Run the test suite |
| `npm run demo:agent` | Guided agent demo (needs ChatGPT/API key) |

---

## 9. Notes on account / AI features

- Graph editing, presets, PyTorch inspection, local execution, and export work **without any
  account or API key**.
- **Ask NeuroBranch** (the AI planner) needs either a desktop **ChatGPT** sign-in
  (Settings → Agent → Continue with ChatGPT) or an **OpenAI API key** (encrypted locally with
  Electron `safeStorage`, never exposed to the renderer).
