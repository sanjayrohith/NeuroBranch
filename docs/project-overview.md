# NeuroBranch — project overview

NeuroBranch is a personal project: an Electron desktop laboratory for designing neural architectures as graphs of small, typed, executable cards. It is an early, evolving MVP rather than a hosted product — everything runs locally on your own machine.

## Motivation

Neural-network code is easy to copy and difficult to inspect. Architectures hide tensor contracts, parallel branches, routing decisions and execution order inside large Python files. NeuroBranch started from a simple question I kept coming back to: what if a model could be assembled, understood and executed as a graph of small typed cards — and what if an assistant had to use the same explicit, auditable tools as a human instead of editing raw code?

## What it does

- Combine more than 100 atomic cards and connect typed "elastic" ports.
- Compare multiple architectures side by side without overwriting the current canvas.
- Inspect synchronized PyTorch generated from the graph.
- Run, rerun or step through a graph with the local atomic player.
- Create reusable, safe PyTorch cards in a dedicated full-canvas studio.
- Save independent workspaces in SQLite, search the catalog in natural language, and export a vector diagram or Python source.

Ask NeuroBranch is a constrained graph planner. It sees the current topology and card capabilities, searches the catalog, creates a safe card when an allowed primitive is missing, connects compatible ports, arranges parallel branches, runs the result, saves it as a workspace and reports any capability it cannot provide. Review mode previews every mutation; Auto apply executes only operations that pass local validation. Existing work can stay read-only while the planner builds a new architecture in parallel.

## How it is built

The desktop shell uses Electron, React, TypeScript and Vite. A typed intermediate representation describes cards, ports, edges, groups and architecture metadata. A semantic registry drives the block library, graph validation, PyTorch generation and the bounded planner tool surface. Topology-aware placement assigns stable execution ranks and parallel lanes while reducing cable crossings. PyTorch execution runs in a separate local Python process through a narrow Electron IPC bridge. Workspaces are stored in a native SQLite database. On desktop, Ask NeuroBranch can use a signed-in ChatGPT account through the bundled OpenAI Codex App Server, and an explicitly configured OpenAI API key is an optional fallback; keys are encrypted with Electron `safeStorage` and are never returned to the renderer.

## Design decisions that shaped it

Keeping four views consistent — the visual graph, tensor contracts, generated PyTorch and runtime execution — was the hardest part. Parallel architectures exposed subtle failure modes: one invalid branch must not leave another waiting forever, layout must preserve forks and joins, and deleting or restoring work must always identify its exact target. So architecture boundaries are explicit, parallel runs are isolated, layout has deterministic tests, and ambiguous destructive actions were replaced with named, confirmed operations.

The broader lesson has been that agentic graph editing works best when the agent is *not* given an unrestricted code editor. A small catalog of observable tools — inspect, search, add, connect, create, move, run, save and export — produces plans that a person can audit and that software can validate. The same typed contracts that make the UI understandable also make agent actions safer.

## Roadmap

- A more self-contained Python runtime.
- More architecture-level validation.
- Reusable compound cards, so a validated subgraph can become one higher-level atomic component.
- Collaborative preset sharing.
- Signed Windows/macOS release automation.

## Built with

Electron, React, TypeScript, Vite, PyTorch, Python, SQLite, Electron `safeStorage`, the OpenAI Codex App Server / OpenAI API, SVG, Vitest, Testing Library, electron-builder.

## Links

- Source: https://github.com/sanjayrohith/NeuroBranch
- Desktop releases: https://github.com/sanjayrohith/NeuroBranch/releases
