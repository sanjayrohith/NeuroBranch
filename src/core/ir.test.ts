import { describe, expect, it } from 'vitest'
import { addNode, compileToPyTorch, createCompositeBlock, duplicateNode, moveGroup, removeNode, updateNodeAttributes, validateGraph } from './ir'
import { gqaPreset } from './presets'
import { deriveGraphStats } from './stats'

describe('Neural Architecture IR', () => {
  it('moves a semantic composite without mutating its child atom positions', () => {
    const moved = moveGroup(gqaPreset, 'qkv-projections', { x: 640, y: 280 })
    expect(moved.groups?.find((group) => group.id === 'qkv-projections')?.position).toEqual({ x: 640, y: 280 })
    expect(moved.nodes).toEqual(gqaPreset.nodes)
    expect(gqaPreset.groups?.find((group) => group.id === 'qkv-projections')?.position).not.toEqual({ x: 640, y: 280 })
  })
  it('compiles the GQA block graph into typed PyTorch SDPA code', () => {
    const validation = validateGraph(gqaPreset)
    const code = compileToPyTorch(gqaPreset)

    expect(validation.valid).toBe(true)
    expect(validation.errors).toEqual([])
    expect(code).toContain('class GeneratedGQA(nn.Module):')
    expect(code).toContain('F.scaled_dot_product_attention(')
    expect(code).toContain('q, expanded_k, expanded_v')
    expect(code).toContain('is_causal=True')
    for (const edge of gqaPreset.edges) {
      expect(code).toContain(`# labo:edge=${edge.id} source=${edge.source} target=${edge.target}`)
    }
  })

  it('changes generated dataflow when an elastic is rewired through a block', () => {
    const withNorm = addNode(gqaPreset, {
      id: 'pre-q-norm', kind: 'custom-pytorch', label: 'Pre-Q RMSNorm', role: 'hidden',
      position: { x: 500, y: 120 }, code: 'nn.RMSNorm(384)',
    })
    const rewired = {
      ...withNorm,
      edges: [
        ...withNorm.edges.filter((edge) => edge.id !== 'hidden-q'),
        { id: 'hidden-pre-q', source: 'hidden', target: 'pre-q-norm' },
        { id: 'pre-q-q', source: 'pre-q-norm', target: 'q-proj' },
      ],
    }
    const code = compileToPyTorch(rewired)

    expect(code).toContain('pre_q_norm = self.pre_q_norm(hidden_states)')
    expect(code).toContain('q_projected = self.q_proj(pre_q_norm)')
    expect(code).not.toContain('q_projected = self.q_proj(hidden_states)')
    expect(code).toContain('# labo:edge=pre-q-q source=pre-q-norm target=q-proj')
  })

  it('does not change PyTorch when only visual XY positions change', () => {
    expect(compileToPyTorch(moveGroup(gqaPreset, 'qkv-projections', { x: 900, y: -300 }))).toBe(compileToPyTorch(gqaPreset))
  })

  it('turns a connected atomic subgraph into a reusable custom block', () => {
    const block = createCompositeBlock(gqaPreset, {
      id: 'qkv-projections',
      label: 'QKV projections',
      nodeIds: ['q-proj', 'k-proj', 'v-proj'],
    })

    expect(block.nodeIds).toEqual(['q-proj', 'k-proj', 'v-proj'])
    expect(block.inputEdges.map((edge) => edge.id)).toEqual(['hidden-q', 'hidden-k', 'hidden-v'])
    expect(block.outputEdges.map((edge) => edge.id)).toEqual(['q-sdpa', 'k-sdpa', 'v-sdpa'])
    expect(block.nodes).toHaveLength(3)
  })

  it('updates one atom without mutating the source graph and recompiles its settings', () => {
    const graph = updateNodeAttributes(gqaPreset, 'q-proj', { bias: true })

    expect(gqaPreset.nodes.find((node) => node.id === 'q-proj')?.attributes?.bias).toBe(false)
    expect(graph.nodes.find((node) => node.id === 'q-proj')?.attributes?.bias).toBe(true)
    expect(deriveGraphStats(graph).parameterCount).toBe(369_024)
    expect(compileToPyTorch(graph)).toContain('self.q_proj = nn.Linear(384, 384, bias=True)')
  })

  it('adds, duplicates, and removes freely composed atoms', () => {
    const added = addNode(gqaPreset, {
      id: 'custom-norm', kind: 'custom-pytorch', label: 'My RMSNorm', role: 'hidden',
      position: { x: 40, y: 40 }, code: 'nn.RMSNorm(384)',
    })
    const duplicated = duplicateNode(added, 'custom-norm', 'custom-norm-copy')
    const removed = removeNode(duplicated, 'custom-norm')

    expect(added.nodes).toHaveLength(gqaPreset.nodes.length + 1)
    expect(duplicated.nodes.some((node) => node.id === 'custom-norm-copy')).toBe(true)
    expect(removed.nodes.some((node) => node.id === 'custom-norm')).toBe(false)
    expect(removed.nodes.some((node) => node.id === 'custom-norm-copy')).toBe(true)
    expect(gqaPreset.nodes.some((node) => node.id === 'custom-norm')).toBe(false)
  })

  it('never recreates a deleted atom from static architecture defaults', () => {
    const graph = removeNode(gqaPreset, 'q-proj')
    const validation = validateGraph(graph)
    const code = compileToPyTorch(graph)

    expect(validation.valid).toBe(false)
    expect(validation.errors).toContain('SDPA requires a connected query input')
    expect(code).not.toContain('self.q_proj')
    expect(code).toContain('Graph is invalid: SDPA requires a connected query input')
  })

  it('emits every added custom atom instead of ignoring it', () => {
    const graph = addNode(gqaPreset, {
      id: 'post-norm', kind: 'custom-pytorch', label: 'Post RMSNorm', role: 'hidden',
      position: { x: 300, y: 430 }, code: 'nn.RMSNorm(384)',
    })
    const code = compileToPyTorch(graph)

    expect(code).toContain('# labo:node=post-norm kind=custom-pytorch')
    expect(code).toContain('self.post_norm = nn.RMSNorm(384)')
  })
})
