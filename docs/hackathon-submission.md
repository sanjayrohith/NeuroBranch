# NeuroBranch — hackathon submission draft

## Project story

### Inspiration

Neural-network code is easy to copy and difficult to inspect. Architectures hide tensor contracts, parallel branches, routing decisions and execution order inside large Python files. NeuroBranch started from a simple question: what if a model could be assembled, understood and executed as a graph of small typed cards—and what if an agent had to use the same explicit tools as a human?

### What it does

NeuroBranch is an Electron desktop laboratory for neural architecture design. Users can combine more than 100 atomic cards, connect typed “elastic” ports, compare multiple architectures side by side, and inspect synchronized PyTorch. The local atomic player can run, rerun or step through a graph. Users can create reusable safe PyTorch cards, save independent workspaces in SQLite, search the catalog in natural language, and export a vector diagram or Python source.

Ask LABO is a constrained graph agent. It sees the current topology and card capabilities, searches the catalog, creates a safe card when an allowed primitive is missing, connects compatible ports, arranges parallel branches, runs the result, saves it as a workspace and reports any capability it cannot provide. Review mode previews every mutation; Auto apply executes only operations that pass local validation. Existing work can remain read-only while the agent builds a new architecture in parallel.

### How we built it

The desktop shell uses Electron, React, TypeScript and Vite. A typed intermediate representation describes cards, ports, edges, groups and architecture metadata. A semantic registry drives the block library, graph validation, PyTorch generation and the bounded agent tool surface. Topology-aware placement assigns stable execution ranks and parallel lanes while reducing cable crossings. PyTorch execution runs in a separate local Python process through a narrow Electron IPC bridge. User workspaces are stored in a native SQLite database; OpenAI API keys are encrypted with Electron `safeStorage` and are never returned to the renderer.

### Challenges

The hardest part was keeping four views consistent: the visual graph, tensor contracts, generated PyTorch and runtime execution. Parallel architectures also exposed subtle failure modes: one invalid branch must not leave another branch waiting forever, layout must preserve forks and joins, and deleting or restoring work must always identify its exact target. We therefore made architecture boundaries explicit, isolated parallel runs, added deterministic layout tests and replaced ambiguous destructive actions with named, confirmed operations.

### What we learned

Agentic graph editing works best when the agent is not given an unrestricted code editor. A small catalog of observable tools—inspect, search, add, connect, create, move, run, save and export—produces plans users can audit and software can validate. The same typed contracts that make the UI understandable also make agent actions safer.

### What is next

Next steps are a self-contained Python runtime, more architecture-level validation, collaborative preset sharing and signed Windows/macOS release automation. We also want reusable compound cards so a validated subgraph can become one higher-level atomic component.

## Built with

Electron, React, TypeScript, Vite, PyTorch, Python, OpenAI API, SQLite, Electron safeStorage, SVG, Vitest, Testing Library, electron-builder

## Try it out

- Source: https://github.com/sanjayrohith/NeuroBranch
- Desktop releases: https://github.com/sanjayrohith/NeuroBranch/releases

## 2–3 minute agent demo

Use a 1440×900 capture, hide notifications and record only the NeuroBranch window. Connect ChatGPT or configure an API key once, then run `npm run demo:agent -- --keep-open`. The automated cues provide the shot order; use `--no-cues` for a clean voice-over-only capture.

| Time | Visual | Voice-over |
| --- | --- | --- |
| 0–8 s | Empty Blank starter in Blocks view | “NeuroBranch turns neural architectures into typed, executable atomic graphs.” |
| 8–20 s | Enter `Hello`; Agent activity returns a conversational answer | “The prompt is conversational, but the agent also understands the live graph and its real capabilities.” |
| 20–38 s | Enter the compact GPT-like QA brief | “I can describe an architecture in plain English instead of manually finding every card.” |
| 38–48 s | Short jump cut while the agent plans | “LABO searches the catalog, chooses atomic cards and wires only compatible typed ports.” |
| 48–68 s | Review graph plan, tool trace, cards and elastics | “The result is an auditable plan. Nothing changes until local validation passes and I approve the complete plan.” |
| 68–82 s | Apply; fitted graph reveals the compact causal decoder | “The first plan is already a valid executable baseline.” |
| 82–105 s | Ask LABO to upgrade the current graph to a token-routed residual MoE | “Now I can iterate: preserve attention and output, but replace the residual MLP with routed and shared experts.” |
| 105–122 s | Review replacements, deletions, new cards and elastics; apply | “LABO edits the existing topology instead of rebuilding blindly.” |
| 122–145 s | Upgraded graph reveals parallel expert paths; atomic player completes | “The topology-aware XY engine keeps forks and joins readable, and this is executable PyTorch—not a drawing.” |
| 145–175 s | PyTorch view, then Split and Agent activity | “Graph, code and the complete agent tool trace share one source of truth.” |

Keep the final edit below 3 minutes, ideally near 2 minutes 40 seconds. If either live model call takes longer, retain only two seconds from each planning wait; do not speed up the reviews, graph construction or execution result.

## Suggested gallery

1. Hero: Split view with a complete routed architecture and generated PyTorch.
2. Ask LABO: Review plan with tools, cards and typed elastics.
3. Parallel comparison: two named architectures on one canvas.
4. Card Builder: category-aware blocks and explicit destination choices.
5. Atomic execution: completed run with generated token output.
