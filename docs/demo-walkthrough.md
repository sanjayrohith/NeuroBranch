# NeuroBranch — agent demo walkthrough

This is a practical script for showing what NeuroBranch and its Ask NeuroBranch planner can do in a couple of minutes. It doubles as the shot list for recording a short screen capture.

## Setup

Connect a ChatGPT account or configure an OpenAI API key once in the app, then run the guided demo:

```bash
npm run demo:agent -- --keep-open
```

The automated cues drive the shot order. Add `--no-cues` for a clean voice-over-only capture. For recording, a 1440×900 window, hidden notifications, and capturing only the NeuroBranch window work well.

## Walkthrough sequence

| Time | Visual | Narration |
| --- | --- | --- |
| 0–8 s | Empty Blank starter in Blocks view | "NeuroBranch turns neural architectures into typed, executable atomic graphs." |
| 8–20 s | Enter `Hello`; Agent activity returns a conversational answer | "The prompt is conversational, but the agent also understands the live graph and its real capabilities." |
| 20–38 s | Enter the compact GPT-like QA brief | "I can describe an architecture in plain English instead of manually finding every card." |
| 38–48 s | Short jump cut while the agent plans | "NeuroBranch searches the catalog, chooses atomic cards and wires only compatible typed ports." |
| 48–68 s | Review graph plan, tool trace, cards and elastics | "The result is an auditable plan. Nothing changes until local validation passes and I approve the complete plan." |
| 68–82 s | Apply; fitted graph reveals the compact causal decoder | "The first plan is already a valid executable baseline." |
| 82–105 s | Ask NeuroBranch to upgrade the current graph to a token-routed residual MoE | "Now I can iterate: preserve attention and output, but replace the residual MLP with routed and shared experts." |
| 105–122 s | Review replacements, deletions, new cards and elastics; apply | "NeuroBranch edits the existing topology instead of rebuilding blindly." |
| 122–145 s | Upgraded graph reveals parallel expert paths; atomic player completes | "The topology-aware XY engine keeps forks and joins readable, and this is executable PyTorch—not a drawing." |
| 145–175 s | PyTorch view, then Split and Agent activity | "Graph, code and the complete agent tool trace share one source of truth." |

Keep the final edit under 3 minutes, ideally near 2 minutes 40 seconds. If either live model call runs long, trim to roughly two seconds of each planning wait; do not speed up the reviews, graph construction or execution result.

## Screens worth capturing

1. Split view with a complete routed architecture and its generated PyTorch.
2. Ask NeuroBranch review plan showing tools, cards and typed elastics.
3. Two named architectures compared side by side on one canvas.
4. Card Builder with category-aware blocks and explicit destination choices.
5. A completed atomic run with generated token output.
