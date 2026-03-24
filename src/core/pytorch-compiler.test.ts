import { describe, expect, it } from 'vitest'
import type { ArchitectureGraph } from './ir'
import { compileToPyTorch, validateGraph } from './ir'
import { compileRegistryGraph, validCustomPyTorchModule } from './pytorch-compiler'
import { complexityDeepPreset, gptLikeStarterPreset, tokenMoePreset, trBasicPreset } from './presets'
import { modelAtomRegistry } from './model-atoms'
import { connectCable, disconnectCable } from './cables'

const normReluGraph = {
  id: 'norm-relu',
  name: 'Norm then ReLU',
  architecture: 'custom',
  config: { hiddenSize: 384, queryHeads: 8, keyValueHeads: 2, headDim: 48 },
  contracts: { causal: false, preservesGqaAtZeroGate: false, sdpaCompatible: false, contextualValue: true },
  nodes: [
    { id: 'hidden', kind: 'input', label: 'Hidden', role: 'hidden', position: { x: 0, y: 0 } },
    { id: 'norm', kind: 'semantic', atomId: 'rms-norm', label: 'RMSNorm', role: 'hidden', position: { x: 0, y: 100 }, attributes: { epsilon: 1e-5 } },
    { id: 'activation', kind: 'semantic', atomId: 'relu', label: 'ReLU', role: 'hidden', position: { x: 0, y: 200 }, attributes: { inplace: false } },
  ],
  edges: [
    { id: 'hidden-norm', source: 'hidden', sourcePort: 'output', target: 'norm', targetPort: 'hidden' },
    { id: 'norm-activation', source: 'norm', sourcePort: 'output', target: 'activation', targetPort: 'hidden' },
  ],
} as unknown as ArchitectureGraph

describe('registry-driven PyTorch compiler', () => {
  it('exposes the basic token-routed residual MLP as a standalone executable graph', () => {
    expect(trBasicPreset.name).toBe('TR Basic · Shared + Residual Top-2')
    expect(trBasicPreset.nodes.map((node) => node.atomId).filter(Boolean)).toEqual([
      'deterministic-token-routing',
      'shared-expert-bank',
      'routed-expert-bank',
      'branch-gated-merge',
    ])
    expect(trBasicPreset.nodes.find((node) => node.id === 'fixed-router')?.attributes).toMatchObject({
      vocabSize: 200019,
      nExperts: 4,
      topK: 2,
      primaryWeight: 0.5,
    })
    expect(trBasicPreset.nodes.find((node) => node.id === 'shared')?.attributes).toMatchObject({
      nSharedExperts: 1,
      intermediateSize: 768,
    })
    expect(trBasicPreset.nodes.find((node) => node.id === 'routed')?.attributes).toMatchObject({
      nExperts: 4,
      intermediateSize: 192,
    })

    const code = compileToPyTorch(trBasicPreset)
    expect(code).toContain('def forward(self, token_ids: torch.Tensor, hidden_states: torch.Tensor):')
    expect(code).toContain("self.register_buffer('fixed_router'")
    expect(code).toContain('self.shared = nn.ModuleList')
    expect(code).toContain('self.routed = nn.ModuleList')
    expect(code).toContain('self.merge_shared_gate = nn.Parameter(torch.tensor(1, dtype=torch.float32))')
    expect(code).toContain('self.merge_routed_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))')
    expect(code).not.toContain('atom=moe-router')
    expect(code).not.toContain('atom=load-balancing-loss')
  })

  it('models TR 300M as standard GQA attention followed by deterministic token-routed MLP', () => {
    const atomIds = complexityDeepPreset.nodes.map((node) => node.atomId).filter(Boolean)
    expect(complexityDeepPreset.config).toMatchObject({ hiddenSize: 1024, queryHeads: 16, keyValueHeads: 4, headDim: 64 })
    expect(atomIds).toEqual(expect.arrayContaining([
      'qkv-projection', 'attention-head-layout', 'qk-normalization', 'rope', 'gqa-kv-expand',
      'causal-sdpa', 'merge-attention-heads', 'attention-output-projection',
      'deterministic-token-routing', 'shared-expert-bank', 'routed-expert-bank', 'branch-gated-merge',
    ]))
    expect(atomIds).not.toContain('moe-router')
    expect(atomIds).not.toContain('top-k-routing')
    const code = compileToPyTorch(complexityDeepPreset)
    expect(code).toContain('F.scaled_dot_product_attention')
    expect(code).toContain('self.attention_output = nn.Linear(16 * 64, 1024, bias=False)')
    expect(code).toContain('self.fixed_routes')
    expect(code).toContain('self.head.weight = self.embedding.weight')
  })

  it('provides a valid dense GPT-like starter without expert routing', () => {
    expect(validateGraph(gptLikeStarterPreset)).toMatchObject({ valid: true, errors: [] })
    expect(gptLikeStarterPreset.config).toMatchObject({ hiddenSize: 768, queryHeads: 12, keyValueHeads: 12, headDim: 64 })
    const atomIds = gptLikeStarterPreset.nodes.map((node) => node.atomId).filter(Boolean)
    expect(atomIds).toContain('swiglu-mlp')
    expect(atomIds).not.toContain('moe-router')
    expect(atomIds).not.toContain('routed-expert-bank')
    const code = compileToPyTorch(gptLikeStarterPreset)
    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).toContain('self.mlp_gate = nn.Linear(768, 3072, bias=False)')
  })

  it('compiles every non-composite Model Blockly definition through its typed ports', () => {
    const compiledIds: string[] = []
    for (const atom of Object.values(modelAtomRegistry)) {
      if (atom.composite) continue
      const inputNodes = atom.inputs.map((port, index) => ({
        id: `source-${index}-${port.id}`,
        kind: 'input' as const,
        label: `${port.id} source`,
        role: port.tensor,
        position: { x: index * 100, y: 0 },
      }))
      const graph = {
        id: `coverage-${atom.id}`,
        name: atom.label,
        architecture: 'custom',
        config: { hiddenSize: 16, queryHeads: 4, keyValueHeads: 2, headDim: 4 },
        contracts: { causal: true, preservesGqaAtZeroGate: false, sdpaCompatible: true, contextualValue: true },
        nodes: [
          ...inputNodes,
          { id: 'under-test', kind: 'semantic' as const, atomId: atom.id, label: atom.label, role: atom.outputs[0]?.tensor ?? 'output', position: { x: 0, y: 100 }, attributes: atom.id === 'lm-head' ? { tieEmbeddingWeights: false } : undefined },
        ],
        edges: atom.inputs.map((port, index) => ({ id: `edge-${index}`, source: inputNodes[index].id, sourcePort: 'output', target: 'under-test', targetPort: port.id })),
      } as unknown as ArchitectureGraph
      const code = compileRegistryGraph(graph)
      expect(code, atom.id).toContain(`# labo:node=under-test atom=${atom.id}`)
      expect(code, `${atom.id} unresolved template`).not.toContain('{{')
      expect(code, `${atom.id} invalid interpolation`).not.toContain('[object Object]')
      compiledIds.push(atom.id)
    }
    expect(compiledIds).toHaveLength(Object.values(modelAtomRegistry).filter((atom) => !atom.composite).length)
    expect(modelAtomRegistry['deepseek-moe'].composite).toBeDefined()
  })

  it('compiles a safe user-created PyTorch card into the semantic graph', () => {
    const graph: ArchitectureGraph = {
      ...normReluGraph,
      id: 'custom-pytorch-card',
      nodes: [
        normReluGraph.nodes[0],
        { id: 'user-linear', kind: 'custom-pytorch', label: 'My projection', role: 'hidden', position: { x: 0, y: 100 }, code: 'nn.Linear(384, 384)' },
      ],
      edges: [{ id: 'hidden-custom', source: 'hidden', sourcePort: 'output', target: 'user-linear', targetPort: 'hidden' }],
    }

    expect(validCustomPyTorchModule('nn.Linear(384, 384)')).toBe(true)
    expect(validCustomPyTorchModule('torch.load("model.pt")')).toBe(false)
    expect(validCustomPyTorchModule("nn.Linear(__import__('os').system('echo nope'), 384)")).toBe(false)
    const code = compileRegistryGraph(graph)
    expect(code).toContain('# labo:node=user-linear kind=custom-pytorch')
    expect(code).toContain('self.user_linear = nn.Linear(384, 384)')
    expect(code).toContain('user_linear_output = self.user_linear(hidden_states)')
  })

  it('compiles a reusable branched card graph with multiple typed inputs as one outer atom', () => {
    const cardGraph: ArchitectureGraph = {
      ...normReluGraph,
      id: 'residual-card',
      nodes: [
        { id: 'residual', kind: 'input', label: 'Residual', role: 'hidden', position: { x: 0, y: 0 } },
        { id: 'branch', kind: 'input', label: 'Branch', role: 'hidden', position: { x: 200, y: 0 } },
        { id: 'merge', kind: 'semantic', atomId: 'residual-add', label: 'Residual add', role: 'hidden', position: { x: 100, y: 120 } },
        { id: 'activation', kind: 'semantic', atomId: 'silu', label: 'SiLU', role: 'hidden', position: { x: 100, y: 240 } },
      ],
      edges: [
        { id: 'residual-merge', source: 'residual', sourcePort: 'hidden', target: 'merge', targetPort: 'residual' },
        { id: 'branch-merge', source: 'branch', sourcePort: 'hidden', target: 'merge', targetPort: 'branch' },
        { id: 'merge-activation', source: 'merge', sourcePort: 'output', target: 'activation', targetPort: 'hidden' },
      ],
    }
    const graph: ArchitectureGraph = {
      ...normReluGraph,
      id: 'outer-composite-card',
      nodes: [
        { id: 'left', kind: 'input', label: 'Left', role: 'hidden', position: { x: 0, y: 0 } },
        { id: 'right', kind: 'input', label: 'Right', role: 'hidden', position: { x: 200, y: 0 } },
        { id: 'card', kind: 'custom-pytorch', label: 'Residual activation', role: 'hidden', position: { x: 100, y: 120 }, code: 'nn.Identity()', customCardGraph: cardGraph },
      ],
      edges: [
        { id: 'left-card', source: 'left', sourcePort: 'hidden', target: 'card', targetPort: 'residual' },
        { id: 'right-card', source: 'right', sourcePort: 'hidden', target: 'card', targetPort: 'branch' },
      ],
    }

    const code = compileRegistryGraph(graph)
    expect(code).toContain('class _LaboCard_card(nn.Module):')
    expect(code).toContain('self.card = _LaboCard_card()')
    expect(code).toContain('card_activation__output = self.card(left, right)')
    expect(code).toContain('return card_activation__output')
  })

  it('uses a neutral token-embedding and MoE graph as the product preset', () => {
    expect(tokenMoePreset.architecture).toBe('custom')
    expect(tokenMoePreset.groups).toBeUndefined()
    expect(tokenMoePreset.nodes.filter((node) => node.kind !== 'input').every((node) => node.kind === 'semantic')).toBe(true)
    expect(tokenMoePreset.nodes.map((node) => node.atomId).filter(Boolean)).toEqual(expect.arrayContaining([
      'token-embedding', 'moe-router', 'top-k-routing', 'routed-expert-bank', 'shared-expert-bank', 'expert-merge',
    ]))
    expect(compileToPyTorch(tokenMoePreset)).toContain('class GeneratedModel(nn.Module):')
  })
  it('compiles semantic atoms and elastic topology without a GQA branch', () => {
    const code = compileRegistryGraph(normReluGraph)

    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).not.toContain('GeneratedGQA')
    expect(code).toContain('# labo:node=norm atom=rms-norm')
    expect(code).toContain('self.norm = nn.RMSNorm(384, eps=1e-05)')
    expect(code).toContain('self.activation = nn.ReLU(inplace=False)')
    expect(code).toContain('# labo:edge=hidden-norm source=hidden target=norm source_port=output target_port=hidden')
    expect(code).toContain('norm_output = self.norm(hidden_states)')
    expect(code).toContain('activation_output = self.activation(norm_output)')
    expect(code).toContain('return activation_output')
  })

  it('rejects rank-3 QKV wired directly into rank-4 attention', () => {
    const invalidAttention: ArchitectureGraph = {
      ...normReluGraph,
      id: 'invalid-attention-ranks',
      nodes: [
        normReluGraph.nodes[0],
        { id: 'qkv', kind: 'semantic', atomId: 'qkv-projection', label: 'QKV', role: 'query', position: { x: 0, y: 100 } },
        { id: 'attention', kind: 'semantic', atomId: 'causal-sdpa', label: 'Attention', role: 'attention', position: { x: 0, y: 200 } },
      ],
      edges: [
        { id: 'hidden-qkv', source: 'hidden', sourcePort: 'output', target: 'qkv', targetPort: 'hidden' },
        { id: 'q', source: 'qkv', sourcePort: 'q', target: 'attention', targetPort: 'q' },
        { id: 'k', source: 'qkv', sourcePort: 'k', target: 'attention', targetPort: 'k' },
        { id: 'v', source: 'qkv', sourcePort: 'v', target: 'attention', targetPort: 'v' },
      ],
    }

    expect(() => compileRegistryGraph(invalidAttention)).toThrow('Rank-3 qkv.q cannot plug into rank-4 attention.q')
    expect(validateGraph(invalidAttention)).toMatchObject({ valid: false, errors: [expect.stringContaining('Rank-3')] })
  })

  it('is the canonical compile path for semantic graphs', () => {
    expect(compileToPyTorch(normReluGraph)).toBe(compileRegistryGraph(normReluGraph))
  })

  it('keeps the block declaration but removes its dataflow call when an elastic is disconnected', () => {
    const disconnected = {
      ...normReluGraph,
      edges: normReluGraph.edges.filter((edge) => edge.target !== 'activation'),
    }
    const code = compileToPyTorch(disconnected)
    expect(() => compileRegistryGraph(disconnected)).toThrow('Missing connected input port hidden on activation')
    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).not.toContain('class GeneratedInvalidGraph')
    expect(code).toContain('self.norm = nn.RMSNorm')
    expect(code).toContain('self.activation = nn.ReLU')
    expect(code).not.toContain('activation_output = self.activation(')
    expect(code).not.toContain('activation_hidden: torch.Tensor')
    expect(code).toContain('return norm_output')
  })

  it('keeps an isolated Blockly card editable as a valid atomic PyTorch draft', () => {
    const isolated: ArchitectureGraph = {
      ...normReluGraph,
      id: 'isolated-relu',
      nodes: [normReluGraph.nodes[2]],
      edges: [],
    }
    const code = compileToPyTorch(isolated)

    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).toContain('# labo:node=activation atom=relu')
    expect(code).toContain('self.activation = nn.ReLU(inplace=False)')
    expect(code).toContain('return None')
    expect(code).not.toContain('GeneratedInvalidGraph')
  })

  it('compiles a standalone Token IDs card as an editable passthrough', () => {
    const inputOnly: ArchitectureGraph = {
      ...normReluGraph,
      id: 'input-only',
      nodes: [{ id: 'token-ids', kind: 'input', label: 'Token IDs', role: 'token-ids', position: { x: 0, y: 0 } }],
      edges: [],
    }
    const code = compileToPyTorch(inputOnly)

    expect(code).toContain('def forward(self, token_ids: torch.Tensor):')
    expect(code).toContain('return token_ids')
    expect(code).not.toContain('GeneratedInvalidGraph')
  })

  it('keeps generated block order stable after disconnecting and reconnecting the same elastic', () => {
    const disconnected = disconnectCable(complexityDeepPreset, 'tokens-embedding')
    const reconnected = connectCable(disconnected, {
      sourceId: 'tokens', sourcePort: 'token-ids', sourcePortId: 'tokenIds',
      targetId: 'embedding', targetPort: 'token-ids', targetPortId: 'tokenIds',
    }).graph
    const nodeOrder = (source: string) => [...source.matchAll(/# labo:node=([^ ]+)/g)].map((match) => match[1])

    expect(nodeOrder(compileToPyTorch(reconnected))).toEqual(nodeOrder(compileToPyTorch(complexityDeepPreset)))
  })

  it('compiles independent architectures with unique inputs and branch-local tied embeddings', () => {
    const graph: ArchitectureGraph = {
      ...normReluGraph,
      nodes: [
        { id: 'tokens-a', kind: 'input', label: 'Tokens A', role: 'token-ids', position: { x: 0, y: 0 } },
        { id: 'embedding-a', kind: 'semantic', atomId: 'token-embedding', label: 'Embedding A', role: 'hidden', position: { x: 0, y: 100 } },
        { id: 'head-a', kind: 'semantic', atomId: 'lm-head', label: 'Head A', role: 'output', position: { x: 0, y: 200 }, attributes: { tieEmbeddingWeights: true } },
        { id: 'tokens-b', kind: 'input', label: 'Tokens B', role: 'token-ids', position: { x: 300, y: 0 } },
        { id: 'embedding-b', kind: 'semantic', atomId: 'token-embedding', label: 'Embedding B', role: 'hidden', position: { x: 300, y: 100 } },
        { id: 'head-b', kind: 'semantic', atomId: 'lm-head', label: 'Head B', role: 'output', position: { x: 300, y: 200 }, attributes: { tieEmbeddingWeights: true } },
      ],
      edges: [
        { id: 'tokens-a-embedding', source: 'tokens-a', sourcePort: 'tokenIds', target: 'embedding-a', targetPort: 'tokenIds' },
        { id: 'embedding-a-head', source: 'embedding-a', sourcePort: 'output', target: 'head-a', targetPort: 'hidden' },
        { id: 'tokens-b-embedding', source: 'tokens-b', sourcePort: 'tokenIds', target: 'embedding-b', targetPort: 'tokenIds' },
        { id: 'embedding-b-head', source: 'embedding-b', sourcePort: 'output', target: 'head-b', targetPort: 'hidden' },
      ],
    }
    const code = compileRegistryGraph(graph)

    expect(code).toContain('def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor):')
    expect(code).toContain('self.head_a.weight = self.embedding_a.weight')
    expect(code).toContain('self.head_b.weight = self.embedding_b.weight')
  })
})
