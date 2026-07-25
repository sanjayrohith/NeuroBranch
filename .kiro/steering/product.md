# Product

NeuroBranch (package name `NeuroBranch`) is a desktop laboratory for composing neural
architectures from executable atomic blocks. It pairs a visual graph editor with
generated PyTorch, step-by-step execution, and a constrained AI graph planner
("Ask NeuroBranch").

## Core ideas

- Build models from 100+ typed, executable "cards" instead of an unrestricted code editor.
- Two-way synchronization between the visual graph and generated PyTorch.
- Deterministic, topology-aware layout that preserves parallel lanes.
- Ask NeuroBranch plans graph changes that are validated locally before mutating the graph;
  review is separate from auto-apply.
- Existing architectures stay read-only in parallel mode; deletion is explicitly targeted.

## Surfaces

- Desktop (Electron) app is the primary target, with ChatGPT sign-in via the Codex
  App Server plus an optional encrypted OpenAI API-key fallback.
- Web build exists but is account/API-key based and does not reuse a browser ChatGPT session.

## Studios

Model, Tokenizer, and Training studios, plus a full-canvas reusable-card creator with
Blocks, PyTorch, and Split views. Local Python runtime executes generated graphs; SVG
diagram and PyTorch/Python export are available.

## Guardrails

- Renderer never receives OpenAI keys after saving (Electron `safeStorage`).
- Custom PyTorch cards accept only supported `torch.nn` constructors and literal args.
- Graphs run through the dedicated local Python runtime, never arbitrary shell eval.
