import { describe, expect, it } from 'vitest'
import { createAgentGraphContext, previewAgentGraphPlan, repairAgentGraphPlan, type AgentGraphPlan } from './agentic-graph'
import { tokenMoePreset } from './presets'

const plan = (connections: AgentGraphPlan['connections'], addedBlocks: AgentGraphPlan['addedBlocks'] = [], createdBlocks: AgentGraphPlan['createdBlocks'] = []): AgentGraphPlan => ({
  summary: 'Connect the existing blocks',
  addedBlocks,
  createdBlocks,
  connections,
  missingBlocks: [],
  warnings: [],
})

describe('agentic graph wiring', () => {
  it('describes semantic blocks and their typed ports', () => {
    const context = createAgentGraphContext(tokenMoePreset)
    const routed = context.graph.nodes.find((node) => node.id === 'routed')

    expect(routed?.inputs).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: 'hidden', tensor: 'hidden' }),
      expect.objectContaining({ id: 'expertIndices', tensor: 'expert-indices' }),
      expect.objectContaining({ id: 'expertWeights', tensor: 'routing-weights' }),
    ]))
    expect(context.availableAtomics.some((atomic) => atomic.atomId === 'top-k-routing')).toBe(true)
    expect(context.availableAtomics).toContainEqual(expect.objectContaining({
      atomId: 'token-ids-input',
      outputs: [expect.objectContaining({ id: 'tokenIds', tensor: 'token-ids', rank: 2 })],
    }))
    expect(context.availableAtomics.find((atomic) => atomic.atomId === 'lm-head')?.settings).toContainEqual(expect.objectContaining({ id: 'tieEmbeddingWeights', default: true }))
  })

  it('exposes raw image, video, audio, and media cards to the agent', () => {
    const context = createAgentGraphContext(tokenMoePreset)
    const ids = new Set(context.availableAtomics.map((atomic) => atomic.atomId))

    for (const atomId of ['image-tensor-input', 'video-tensor-input', 'audio-waveform-input', 'image-channel-normalization', 'image-patch-embedding', 'video-channel-normalization', 'video-tubelet-embedding', 'audio-waveform-normalization', 'audio-frame-embedding']) {
      expect(ids.has(atomId), `missing ${atomId} from agent catalog`).toBe(true)
    }
  })

  it('exposes saved custom cards to the agent search context', () => {
    const context = createAgentGraphContext(tokenMoePreset, 'extend', [{ id: 'my-gelu', label: 'My GELU', code: 'nn.GELU()', inputRole: 'hidden', outputRole: 'hidden' }])
    expect(context.availableCustomCards).toEqual([{ id: 'my-gelu', label: 'My GELU', code: 'nn.GELU()', inputRole: 'hidden', outputRole: 'hidden' }])
  })

  it('describes whole architectures so the agent can clean them as one unit', () => {
    const context = createAgentGraphContext(tokenMoePreset)
    expect(context.architectures).toHaveLength(1)
    expect(context.architectures[0]).toMatchObject({ id: 'architecture-1-tokens', label: tokenMoePreset.name })
    expect(new Set(context.architectures[0]?.nodeIds)).toEqual(new Set(tokenMoePreset.nodes.map((node) => node.id)))
  })

  it('lets the agent add a typed Token IDs source on a blank graph', () => {
    const graph = { ...tokenMoePreset, nodes: [], edges: [], groups: [] }
    const preview = previewAgentGraphPlan(graph, plan([{
      sourceId: 'tokens', sourcePortId: 'tokenIds', targetId: 'embedding', targetPortId: 'tokenIds', reason: 'Feed tokens to embedding',
    }], [
      { atomId: 'token-ids-input', nodeId: 'tokens', reason: 'Blank graph needs an input source' },
      { atomId: 'token-embedding', nodeId: 'embedding', reason: 'Embed input tokens' },
    ]))

    expect(preview.acceptedBlocks).toHaveLength(2)
    expect(preview.accepted).toHaveLength(1)
    expect(preview.graph.nodes).toContainEqual(expect.objectContaining({ id: 'tokens', kind: 'input', role: 'token-ids' }))
    expect(preview.graph.edges).toContainEqual(expect.objectContaining({ source: 'tokens', sourcePort: 'tokenIds', target: 'embedding', targetPort: 'tokenIds' }))
  })

  it('normalizes underscore IDs across a complete multimodal plan before validation', () => {
    const graph = { ...tokenMoePreset, nodes: [], edges: [], groups: [] }
    const repaired = repairAgentGraphPlan(graph, plan([
      { sourceId: 'text_tokens', sourcePortId: 'tokenIds', targetId: 'text_embedding', targetPortId: 'tokenIds', reason: 'Embed text' },
      { sourceId: 'image_input', sourcePortId: 'image', targetId: 'image_patch_embedding', targetPortId: 'image', reason: 'Embed image patches' },
    ], [
      { atomId: 'token-ids-input', nodeId: 'text_tokens', reason: 'Text source' },
      { atomId: 'token-embedding', nodeId: 'text_embedding', reason: 'Text embedding' },
      { atomId: 'image-tensor-input', nodeId: 'image_input', reason: 'Image source' },
      { atomId: 'image-patch-embedding', nodeId: 'image_patch_embedding', reason: 'Image patches' },
    ]))
    const preview = previewAgentGraphPlan(graph, repaired)

    expect(repaired.addedBlocks.map((block) => block.nodeId)).toEqual(['text-tokens', 'text-embedding', 'image-input', 'image-patch-embedding'])
    expect(repaired.connections).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceId: 'text-tokens', targetId: 'text-embedding' }),
      expect.objectContaining({ sourceId: 'image-input', targetId: 'image-patch-embedding' }),
    ]))
    expect(preview.acceptedBlocks).toHaveLength(4)
    expect(preview.accepted).toHaveLength(2)
    expect(preview.rejectedBlocks).toHaveLength(0)
    expect(preview.rejected).toHaveLength(0)
  })

  it('previews a valid missing elastic without mutating the original graph', () => {
    const graph = { ...tokenMoePreset, edges: tokenMoePreset.edges.filter((edge) => edge.id !== 'router-topk') }
    const preview = previewAgentGraphPlan(graph, plan([{
      sourceId: 'router', sourcePortId: 'scores', targetId: 'topk', targetPortId: 'scores', reason: 'Route scores into selection',
    }]))

    expect(graph.edges).toHaveLength(10)
    expect(preview.accepted).toHaveLength(1)
    expect(preview.graph.edges).toHaveLength(11)
    expect(preview.graph.edges).toContainEqual(expect.objectContaining({ source: 'router', sourcePort: 'scores', target: 'topk', targetPort: 'scores' }))
  })

  it('adds a library atomic before wiring it to an existing block', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, plan([{
      sourceId: 'norm', sourcePortId: 'output', targetId: 'agent-relu', targetPortId: 'hidden', reason: 'Activate normalized hidden states',
    }], [{ atomId: 'relu', nodeId: 'agent-relu', reason: 'The requested activation is not on the canvas' }]))

    expect(tokenMoePreset.nodes.some((node) => node.id === 'agent-relu')).toBe(false)
    expect(preview.acceptedBlocks).toHaveLength(1)
    expect(preview.accepted).toHaveLength(1)
    expect(preview.graph.nodes).toContainEqual(expect.objectContaining({ id: 'agent-relu', atomId: 'relu', kind: 'semantic' }))
    expect(preview.graph.edges).toContainEqual(expect.objectContaining({ source: 'norm', target: 'agent-relu', targetPort: 'hidden' }))
  })

  it('lays out agent-created parallel branches by graph depth', () => {
    const graph = { ...tokenMoePreset, nodes: [], edges: [], groups: [] }
    const preview = previewAgentGraphPlan(graph, plan([
      { sourceId: 'input-node', sourcePortId: 'output', targetId: 'left-branch', targetPortId: 'hidden', reason: 'Left branch' },
      { sourceId: 'input-node', sourcePortId: 'output', targetId: 'right-branch', targetPortId: 'hidden', reason: 'Right branch' },
    ], [
      { atomId: 'identity', nodeId: 'input-node', reason: 'Input transform' },
      { atomId: 'relu', nodeId: 'left-branch', reason: 'Parallel left' },
      { atomId: 'gelu', nodeId: 'right-branch', reason: 'Parallel right' },
    ]))
    const input = preview.graph.nodes.find((node) => node.id === 'input-node')!
    const left = preview.graph.nodes.find((node) => node.id === 'left-branch')!
    const right = preview.graph.nodes.find((node) => node.id === 'right-branch')!

    expect(left.position.y).toBe(right.position.y)
    expect(left.position.x).not.toBe(right.position.x)
    expect(input.position.y).toBeLessThan(left.position.y)
  })

  it('creates a safe unary PyTorch card and wires its hidden ports', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, plan([{
      sourceId: 'norm', sourcePortId: 'output', targetId: 'agent-projection', targetPortId: 'hidden', reason: 'Project normalized states',
    }], [], [{
      nodeId: 'agent-projection', label: 'Agent projection', pytorchModule: 'nn.Linear(384, 384, bias=False)', reason: 'No exact library card was selected',
    }]))

    expect(preview.acceptedCreatedBlocks).toHaveLength(1)
    expect(preview.graph.nodes).toContainEqual(expect.objectContaining({ id: 'agent-projection', kind: 'custom-pytorch', code: 'nn.Linear(384, 384, bias=False)' }))
    expect(preview.graph.edges).toContainEqual(expect.objectContaining({ source: 'norm', target: 'agent-projection', targetPort: 'hidden' }))
  })

  it('rejects unsafe generated PyTorch cards before they reach the graph', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, plan([], [], [{
      nodeId: 'unsafe-card', label: 'Unsafe card', pytorchModule: 'torch.load("weights.pt")', reason: 'Unsafe request',
    }]))

    expect(preview.acceptedCreatedBlocks).toHaveLength(0)
    expect(preview.rejectedBlocks[0]?.reason).toContain('safe nn.Module subset')
    expect(preview.graph.nodes.some((node) => node.id === 'unsafe-card')).toBe(false)
  })

  it('rejects unavailable, composite, and duplicate agent blocks', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, plan([], [
      { atomId: 'not-in-library', nodeId: 'unknown-atomic', reason: 'Unavailable' },
      { atomId: 'deepseek-moe', nodeId: 'composite-recipe', reason: 'Composite' },
      { atomId: 'rms-norm', nodeId: 'norm', reason: 'Duplicate id' },
    ]))

    expect(preview.acceptedBlocks).toHaveLength(0)
    expect(preview.rejectedBlocks.map(({ reason }) => reason)).toEqual([
      expect.stringContaining('not available'),
      expect.stringContaining('Composite recipes'),
      expect.stringContaining('already exists'),
    ])
  })

  it('rejects occupied, incompatible, and cyclic connections', () => {
    const occupied = previewAgentGraphPlan(tokenMoePreset, plan([{
      sourceId: 'router', sourcePortId: 'scores', targetId: 'topk', targetPortId: 'scores', reason: 'Duplicate',
    }]))
    const incompatible = previewAgentGraphPlan({ ...tokenMoePreset, edges: [] }, plan([{
      sourceId: 'router', sourcePortId: 'scores', targetId: 'head', targetPortId: 'hidden', reason: 'Wrong tensor',
    }]))
    const cyclic = previewAgentGraphPlan({ ...tokenMoePreset, edges: [] }, plan([
      { sourceId: 'norm', sourcePortId: 'output', targetId: 'shared', targetPortId: 'hidden', reason: 'Forward edge' },
      { sourceId: 'shared', sourcePortId: 'output', targetId: 'norm', targetPortId: 'hidden', reason: 'Cycle attempt' },
    ]))

    expect(occupied.rejected[0]?.reason).toContain('already connected')
    expect(incompatible.rejected[0]?.reason).toContain('cannot plug')
    expect(cyclic.accepted).toHaveLength(1)
    expect(cyclic.rejected[0]?.reason).toContain('cycle')
  })

  it('builds a separate architecture without changing or connecting to existing work', () => {
    const originalNodes = tokenMoePreset.nodes.map((node) => ({ ...node, position: { ...node.position } }))
    const originalEdges = tokenMoePreset.edges.map((edge) => ({ ...edge }))
    const preview = previewAgentGraphPlan(tokenMoePreset, plan([
      { sourceId: 'parallel-input', sourcePortId: 'hidden', targetId: 'parallel-relu', targetPortId: 'hidden', reason: 'New branch' },
      { sourceId: 'norm', sourcePortId: 'output', targetId: 'parallel-relu', targetPortId: 'hidden', reason: 'Must not touch existing work' },
    ], [
      { atomId: 'hidden-state-input', nodeId: 'parallel-input', reason: 'Independent source' },
      { atomId: 'relu', nodeId: 'parallel-relu', reason: 'Independent model' },
    ]), 'parallel')

    expect(preview.accepted).toHaveLength(1)
    expect(preview.rejected[0]?.reason).toContain('cannot connect to or modify')
    expect(preview.graph.nodes.slice(0, originalNodes.length)).toEqual(originalNodes)
    expect(preview.graph.edges.slice(0, originalEdges.length)).toEqual(originalEdges)
    expect(preview.graph.nodes.find((node) => node.id === 'parallel-input')!.position.x).toBeGreaterThan(Math.max(...originalNodes.map((node) => node.position.x)))
  })

  it('previews edits, deletion, movement and queued actions without mutating the source graph', () => {
    const sourceNorm = tokenMoePreset.nodes.find((node) => node.id === 'norm')!
    const preview = previewAgentGraphPlan(tokenMoePreset, {
      ...plan([]),
      updatedBlocks: [{ nodeId: 'norm', label: 'Agent RMSNorm', settings: { eps: 0.00001 }, pytorchModule: null, reason: 'Tune norm' }],
      deletedBlocks: [{ nodeId: 'router-aux-loss', reason: 'Remove objective' }],
      movedBlocks: [{ nodeId: 'norm', x: 123, y: 456, reason: 'Exact placement' }],
      actions: [{ type: 'run', mode: 'step', reason: 'Verify one atom' }, { type: 'export', kind: 'svg', reason: 'Share graph' }],
    })

    expect(tokenMoePreset.nodes.find((node) => node.id === 'norm')).toEqual(sourceNorm)
    expect(preview.graph.nodes.find((node) => node.id === 'norm')).toMatchObject({ label: 'Agent RMSNorm', position: { x: 123, y: 456 }, attributes: { eps: 0.00001 } })
    expect(preview.graph.nodes.some((node) => node.id === 'router-aux-loss')).toBe(false)
    expect(preview.acceptedActions).toEqual(expect.arrayContaining([expect.objectContaining({ type: 'run' }), expect.objectContaining({ type: 'export' })]))
  })

  it('keeps Edit Cards restricted to the active selection and never adds cards', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, {
      ...plan([], [{ atomId: 'relu', nodeId: 'new-relu', reason: 'Must use Add Blocks' }]),
      updatedBlocks: [
        { nodeId: 'norm', label: 'Selected norm', settings: null, pytorchModule: null, reason: 'Selected edit' },
        { nodeId: 'shared', label: 'Outside selection', settings: null, pytorchModule: null, reason: 'Forbidden edit' },
      ],
      movedBlocks: [{ nodeId: 'shared', x: 0, y: 0, reason: 'Forbidden move' }],
      actions: [{ type: 'layout', scope: 'all', reason: 'Forbidden relayout' }],
    }, 'extend', { editingNodeIds: ['norm'] })

    expect(preview.acceptedBlocks).toHaveLength(0)
    expect(preview.rejectedBlocks[0]?.reason).toContain('switch to Add Blocks')
    expect(preview.graph.nodes.find((node) => node.id === 'norm')?.label).toBe('Selected norm')
    expect(preview.graph.nodes.find((node) => node.id === 'shared')?.label).not.toBe('Outside selection')
    expect(preview.rejectedMutations.map((mutation) => mutation.reason)).toEqual(expect.arrayContaining([
      expect.stringContaining('active selection'),
      expect.stringContaining('move only cards'),
      expect.stringContaining('cannot relayout'),
    ]))
  })

  it('rejects mutations of existing cards in parallel mode', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, {
      ...plan([]),
      updatedBlocks: [{ nodeId: 'norm', label: 'Changed', settings: null, pytorchModule: null, reason: 'Forbidden edit' }],
      deletedBlocks: [{ nodeId: 'shared', reason: 'Forbidden deletion' }],
      movedBlocks: [{ nodeId: 'routed', x: 0, y: 0, reason: 'Forbidden move' }],
      actions: [{ type: 'layout', scope: 'all', reason: 'Forbidden global layout' }],
    }, 'parallel')

    expect(preview.rejectedMutations).toHaveLength(4)
    expect(preview.graph.nodes.find((node) => node.id === 'norm')?.label).toBe(tokenMoePreset.nodes.find((node) => node.id === 'norm')?.label)
    expect(preview.graph.nodes.some((node) => node.id === 'shared')).toBe(true)
  })

  it('rejects per-card edits of graph-wide model dimensions', () => {
    const preview = previewAgentGraphPlan(tokenMoePreset, {
      ...plan([]),
      updatedBlocks: [{ nodeId: 'embedding', label: 'Embedding 256', settings: { hiddenSize: 256 }, pytorchModule: null, reason: 'Shrink one card' }],
    })

    expect(preview.rejectedMutations).toEqual([expect.objectContaining({ reason: expect.stringContaining('graph-wide dimensions') })])
    expect(preview.graph.nodes.find((node) => node.id === 'embedding')?.attributes?.hiddenSize).not.toBe(256)
  })

  it('repairs false missing Token IDs and logits-decoder claims with native cards', () => {
    const graph = { ...tokenMoePreset, nodes: [], edges: [], groups: [] }
    const repaired = repairAgentGraphPlan(graph, {
      summary: 'Language model',
      addedBlocks: [
        { atomId: 'token-embedding', nodeId: 'embedding', reason: 'Embed tokens' },
        { atomId: 'lm-head', nodeId: 'head', reason: 'Create logits' },
      ],
      createdBlocks: [],
      connections: [{ sourceId: 'embedding', sourcePortId: 'output', targetId: 'head', targetPortId: 'hidden', reason: 'Project states' }],
      missingBlocks: [
        { atomId: null, label: 'Source de Token IDs / tokenizer', reason: 'Aucun atomic ne fournit token-ids.' },
        { atomId: null, label: 'Échantillonneur ou décodeur de logits', reason: 'Convertir les logits en token généré.' },
      ],
      warnings: [],
    })

    expect(repaired.addedBlocks.map((block) => block.atomId)).toEqual(expect.arrayContaining(['token-ids-input', 'greedy-token-decoder']))
    expect(repaired.connections).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourcePortId: 'tokenIds', targetId: 'embedding', targetPortId: 'tokenIds' }),
      expect.objectContaining({ sourceId: 'head', sourcePortId: 'logits', targetPortId: 'logits' }),
    ]))
    expect(repaired.missingBlocks).toEqual([])
    expect(previewAgentGraphPlan(graph, repaired).rejected).toEqual([])
  })

  it('inserts the required head layout between rank-3 QKV and rank-4 attention', () => {
    const graph = { ...tokenMoePreset, nodes: [], edges: [], groups: [] }
    const repaired = repairAgentGraphPlan(graph, plan([
      { sourceId: 'qkv', sourcePortId: 'q', targetId: 'attention', targetPortId: 'q', reason: 'Direct Q' },
      { sourceId: 'qkv', sourcePortId: 'k', targetId: 'attention', targetPortId: 'k', reason: 'Direct K' },
      { sourceId: 'qkv', sourcePortId: 'v', targetId: 'attention', targetPortId: 'v', reason: 'Direct V' },
    ], [
      { atomId: 'qkv-projection', nodeId: 'qkv', reason: 'Project QKV' },
      { atomId: 'causal-sdpa', nodeId: 'attention', reason: 'Attend causally' },
    ]))

    const layout = repaired.addedBlocks.find((block) => block.atomId === 'attention-head-layout')
    expect(layout).toBeDefined()
    expect(repaired.connections.some((connection) => connection.sourceId === 'qkv' && connection.targetId === 'attention')).toBe(false)
    expect(repaired.connections).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceId: 'qkv', sourcePortId: 'q', targetId: layout?.nodeId, targetPortId: 'q' }),
      expect.objectContaining({ sourceId: layout?.nodeId, sourcePortId: 'qHeads', targetId: 'attention', targetPortId: 'q' }),
    ]))
    expect(previewAgentGraphPlan(graph, repaired).rejected).toEqual([])
  })
})
