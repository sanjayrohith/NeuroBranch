import { execFileSync } from 'node:child_process'
import { describe, expect, it } from 'vitest'
import { compileToPyTorch } from '../src/core/ir'
import type { ArchitectureGraph } from '../src/core/ir'
import { compileRegistryGraph } from '../src/core/pytorch-compiler'
import { complexityDeepPreset, gqaPreset, trBasicPreset } from '../src/core/presets'
import { modelAtomRegistry, type AtomPort } from '../src/core/model-atoms'

function runtimeInput(atomId: string, port: AtomPort): string {
  if (atomId === 'attention-output-projection') return 'torch.randn(1, 2, 8)'
  if (atomId === 'attention-value-mix' && port.tensor === 'attention') return 'torch.randn(1, 2, 2, 2)'
  if (atomId === 'binary-cross-entropy' && port.id === 'targets') return 'torch.rand(1, 2, 16)'
  if (atomId === 'media-token-selector' && port.id === 'indices') return 'torch.randint(0, 2, (1, 2))'
  if (port.tensor === 'image') return 'torch.randn(1, 3, 32, 32)'
  if (port.tensor === 'video') return 'torch.randn(1, 3, 4, 32, 32)'
  if (port.tensor === 'audio') return 'torch.randn(1, 1, 3200)'
  if (port.tensor === 'token-ids') return 'torch.randint(0, 64, (1, 2))'
  if (port.tensor === 'labels') return 'torch.randint(0, 32, (1, 2))'
  if (port.tensor === 'logits') return 'torch.randn(1, 2, 32)'
  if (port.tensor === 'routing-logits') return 'torch.randn(1, 2, 64)'
  if (port.tensor === 'expert-indices') return 'torch.randint(0, 64, (1, 2, 6))'
  if (port.tensor === 'routing-weights') return 'torch.softmax(torch.randn(1, 2, 6), dim=-1)'
  if (port.tensor === 'scalar') return 'torch.tensor(0.5)'
  if (port.tensor === 'query' || port.tensor === 'key' || port.tensor === 'value') {
    return atomId === 'attention-head-layout' ? 'torch.randn(1, 2, 8)' : 'torch.randn(1, 2, 2, 4)'
  }
  if (port.tensor === 'attention') return 'torch.randn(1, 2, 2, 4)'
  return 'torch.randn(1, 2, 16)'
}

describe('generated PyTorch block graph', () => {
  it('executes TR Basic from token IDs and contextual hidden states with real torch', () => {
    const graph = {
      ...structuredClone(trBasicPreset),
      config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 1, headDim: 8 },
      nodes: trBasicPreset.nodes.map((node) => {
        if (node.id === 'fixed-router') return { ...node, attributes: { vocabSize: 64, nExperts: 4, topK: 2, layerIndex: 0, primaryWeight: 0.5 } }
        if (node.id === 'shared') return { ...node, attributes: { nSharedExperts: 1, intermediateSize: 32, activation: 'swiglu' } }
        if (node.id === 'routed') return { ...node, attributes: { nExperts: 4, intermediateSize: 8, activation: 'swiglu', expertParallelSize: 1 } }
        return node
      }),
    } as ArchitectureGraph
    const generated = compileRegistryGraph(graph)
    const smoke = `${generated}\nmodel = GeneratedModel()\ntokens = torch.randint(0, 64, (2, 5))\nhidden = torch.randn(2, 5, 16)\nwith torch.no_grad():\n    output = model(tokens, hidden)\nprint(tuple(output.shape), tuple(model.fixed_router.shape), model.fixed_router.dtype)\n`
    const output = execFileSync('.venv/bin/python', ['-c', smoke], { cwd: process.cwd(), encoding: 'utf8' })
    expect(output.trim()).toBe('(2, 5, 16) (2, 64) torch.int64')
  })

  it('executes the canonical TR 300M attention plus deterministic residual MLP graph', () => {
    const graph = {
      ...structuredClone(complexityDeepPreset),
      config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 1, headDim: 8 },
      nodes: complexityDeepPreset.nodes.map((node) => {
        if (node.id === 'embedding') return { ...node, attributes: { vocabSize: 64, hiddenSize: 16 } }
        if (node.id === 'head-layout') return { ...node, attributes: { queryHeads: 2, keyValueHeads: 1, headDim: 8 } }
        if (node.id === 'fixed-routes') return { ...node, attributes: { vocabSize: 64, nExperts: 4, topK: 2, layerIndex: 0, primaryWeight: 0.5 } }
        if (node.id === 'shared') return { ...node, attributes: { nSharedExperts: 1, intermediateSize: 32, activation: 'swiglu' } }
        if (node.id === 'routed') return { ...node, attributes: { nExperts: 4, intermediateSize: 8, activation: 'swiglu', expertParallelSize: 1 } }
        if (node.id === 'head') return { ...node, attributes: { vocabSize: 64, tieEmbeddingWeights: true, bias: false } }
        return node
      }),
    } as ArchitectureGraph
    const generated = compileRegistryGraph(graph)
    const smoke = `${generated}\nmodel = GeneratedModel()\ntokens = torch.randint(0, 64, (2, 5))\nwith torch.no_grad():\n    logits = model(tokens)\nprint(tuple(logits.shape), model.head.weight is model.embedding.weight)\n`
    const output = execFileSync('.venv/bin/python', ['-c', smoke], { cwd: process.cwd(), encoding: 'utf8' })
    expect(output.trim()).toBe('(2, 5, 64) True')
  })

  it('executes every non-composite Model atom with real torch tensors', () => {
    const scripts: string[] = ['import gc']
    const expected: string[] = []
    for (const atom of Object.values(modelAtomRegistry)) {
      if (atom.composite) continue
      const inputs = atom.inputs.map((port, index) => ({ id: `source-${index}-${port.id}`, kind: 'input', label: port.id, role: port.tensor, position: { x: index, y: 0 } }))
      const graph = {
        id: `runtime-${atom.id}`, name: atom.label, architecture: 'custom',
        config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 2, headDim: 4 },
        contracts: { causal: true, preservesGqaAtZeroGate: false, sdpaCompatible: true, contextualValue: true },
        nodes: [...inputs, { id: 'under-test', kind: 'semantic', atomId: atom.id, label: atom.label, role: atom.outputs[0]?.tensor ?? 'output', position: { x: 0, y: 1 }, attributes: atom.id === 'lm-head' ? { tieEmbeddingWeights: false } : undefined }],
        edges: atom.inputs.map((port, index) => ({ id: `edge-${index}`, source: inputs[index].id, sourcePort: 'output', target: 'under-test', targetPort: port.id })),
      } as unknown as ArchitectureGraph
      scripts.push(compileRegistryGraph(graph))
      scripts.push(`model = GeneratedModel()\nwith torch.no_grad():\n    model(${atom.inputs.map((port) => runtimeInput(atom.id, port)).join(', ')})\nprint(${JSON.stringify(atom.id)})\ndel model\ngc.collect()`)
      expected.push(atom.id)
    }
    const output = execFileSync('.venv/bin/python', ['-c', scripts.join('\n\n')], { cwd: process.cwd(), encoding: 'utf8', maxBuffer: 1024 * 1024 })
    expect(output.trim().split('\n')).toEqual(expected)
  }, 120_000)

  it('executes the compiled GQA block and elastic dataflow with real torch', () => {
    const generated = compileToPyTorch(gqaPreset)
    const smoke = `${generated}\nmodel = GeneratedGQA()\nx = torch.randn(2, 7, 384)\ny = model(x)\nprint(tuple(y.shape))\n`
    const output = execFileSync('.venv/bin/python', ['-c', smoke], {
      cwd: process.cwd(),
      encoding: 'utf8',
    })
    expect(output.trim()).toBe('(2, 7, 384)')
  })

  it('executes token embedding with routed and shared experts from semantic blocks', () => {
    const graph = {
      id: 'token-moe', name: 'Token MoE', architecture: 'custom',
      config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 2, headDim: 8 },
      contracts: { causal: false, preservesGqaAtZeroGate: false, sdpaCompatible: false, contextualValue: true },
      nodes: [
        { id: 'tokens', kind: 'input', label: 'Token IDs', role: 'token-ids', position: { x: 0, y: 0 } },
        { id: 'embedding', kind: 'semantic', atomId: 'token-embedding', label: 'Token embedding', role: 'hidden', position: { x: 0, y: 1 }, attributes: { vocabSize: 128, hiddenSize: 16 } },
        { id: 'norm', kind: 'semantic', atomId: 'rms-norm', label: 'RMSNorm', role: 'hidden', position: { x: 0, y: 2 } },
        { id: 'router', kind: 'semantic', atomId: 'moe-router', label: 'Router', role: 'output', position: { x: 0, y: 3 }, attributes: { nExperts: 4, scoringFunction: 'softmax' } },
        { id: 'topk', kind: 'semantic', atomId: 'top-k-routing', label: 'Top K', role: 'output', position: { x: 0, y: 4 }, attributes: { topK: 2, nExpertGroups: 2, topkGroups: 1 } },
        { id: 'routed', kind: 'semantic', atomId: 'routed-expert-bank', label: 'Routed experts', role: 'hidden', position: { x: 0, y: 5 }, attributes: { nExperts: 4, intermediateSize: 32 } },
        { id: 'shared', kind: 'semantic', atomId: 'shared-expert-bank', label: 'Shared experts', role: 'hidden', position: { x: 0, y: 6 }, attributes: { nSharedExperts: 1, intermediateSize: 32 } },
        { id: 'merge', kind: 'semantic', atomId: 'expert-merge', label: 'Merge', role: 'hidden', position: { x: 0, y: 7 } },
        { id: 'head', kind: 'semantic', atomId: 'lm-head', label: 'LM head', role: 'output', position: { x: 0, y: 8 }, attributes: { vocabSize: 128, tieEmbeddingWeights: false } },
      ],
      edges: [
        { id: 'tokens-embedding', source: 'tokens', sourcePort: 'tokenIds', target: 'embedding', targetPort: 'tokenIds' },
        { id: 'embedding-norm', source: 'embedding', sourcePort: 'output', target: 'norm', targetPort: 'hidden' },
        { id: 'norm-router', source: 'norm', sourcePort: 'output', target: 'router', targetPort: 'hidden' },
        { id: 'router-topk', source: 'router', sourcePort: 'scores', target: 'topk', targetPort: 'scores' },
        { id: 'norm-routed', source: 'norm', sourcePort: 'output', target: 'routed', targetPort: 'hidden' },
        { id: 'topk-indices', source: 'topk', sourcePort: 'expertIndices', target: 'routed', targetPort: 'expertIndices' },
        { id: 'topk-weights', source: 'topk', sourcePort: 'expertWeights', target: 'routed', targetPort: 'expertWeights' },
        { id: 'norm-shared', source: 'norm', sourcePort: 'output', target: 'shared', targetPort: 'hidden' },
        { id: 'routed-merge', source: 'routed', sourcePort: 'output', target: 'merge', targetPort: 'routed' },
        { id: 'shared-merge', source: 'shared', sourcePort: 'output', target: 'merge', targetPort: 'shared' },
        { id: 'merge-head', source: 'merge', sourcePort: 'output', target: 'head', targetPort: 'hidden' },
      ],
    } as unknown as ArchitectureGraph
    const generated = compileRegistryGraph(graph)
    const smoke = `${generated}\nmodel = GeneratedModel()\ntokens = torch.randint(0, 128, (2, 5))\ny = model(tokens)\nprint(tuple(y.shape))\n`
    const output = execFileSync('.venv/bin/python', ['-c', smoke], { cwd: process.cwd(), encoding: 'utf8' })
    expect(output.trim()).toBe('(2, 5, 128)')
  })
})
