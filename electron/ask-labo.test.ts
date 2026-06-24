import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('./openai-credentials', () => ({
  resolveOpenAIConfig: async () => process.env.OPENAI_API_KEY
    ? { apiKey: process.env.OPENAI_API_KEY, model: process.env.OPENAI_MODEL || 'gpt-5.6-terra' }
    : undefined,
}))

import { askLabo, validateAskLaboPayload } from './ask-labo'

const previousApiKey = process.env.OPENAI_API_KEY
const previousModel = process.env.OPENAI_MODEL
const functionCall = (name: string, arguments_: Record<string, unknown>, callId: string) => ({ type: 'function_call', name, arguments: JSON.stringify(arguments_), call_id: callId })

afterEach(() => {
  vi.unstubAllGlobals()
  if (previousApiKey === undefined) delete process.env.OPENAI_API_KEY
  else process.env.OPENAI_API_KEY = previousApiKey
  if (previousModel === undefined) delete process.env.OPENAI_MODEL
  else process.env.OPENAI_MODEL = previousModel
})

describe('Ask LABO OpenAI bridge', () => {
  it('rejects empty and malformed renderer payloads', () => {
    expect(() => validateAskLaboPayload({ request: '', context: {} })).toThrow('cannot be empty')
    expect(() => validateAskLaboPayload(null as never)).toThrow('requires a request')
  })

  it('runs a bounded function-calling loop without exposing the API key', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    process.env.OPENAI_MODEL = 'test-model'
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestBody = JSON.parse(String(init?.body)) as { model: string; tools: Array<{ name: string; strict: boolean }>; input: unknown[]; parallel_tool_calls: boolean }
      expect(requestBody.model).toBe('test-model')
      expect(requestBody.parallel_tool_calls).toBe(true)
      expect(requestBody.tools).toEqual(expect.arrayContaining([expect.objectContaining({ name: 'search_cards', strict: true }), expect.objectContaining({ name: 'finish_plan', strict: true })]))
      expect(init?.headers).toMatchObject({ Authorization: 'Bearer test-secret-key' })
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({ output: [functionCall('add_block', { atom_id: 'relu', node_id: 'agent-relu', reason: 'Activation' }, 'call-add')] }), { status: 200 })
      expect(requestBody.input).toEqual(expect.arrayContaining([expect.objectContaining({ type: 'function_call_output', call_id: 'call-add' })]))
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Build it', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({ request: 'Wire my blocks', context: { graph: { nodes: [] }, availableAtomics: [{ atomId: 'relu', label: 'ReLU', inputs: [], outputs: [{ id: 'output', tensor: 'hidden' }] }] } })
    expect(result).toMatchObject({ summary: 'Build it', addedBlocks: [{ atomId: 'relu', nodeId: 'agent-relu', reason: 'Activation' }] })
    expect(result.toolTrace.map((item) => item.tool)).toEqual(['add_block', 'finish_plan'])
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('requires a main-process API key', async () => {
    delete process.env.OPENAI_API_KEY
    await expect(askLabo({ request: 'Wire my blocks', context: {} })).rejects.toThrow('No OpenAI API key')
  })

  it('gives the agent the Create card auto-composer for a missing capability', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestBody = JSON.parse(String(init?.body)) as { tools: Array<{ name: string }>; input: Array<{ type?: string; output?: string }>; instructions: string }
      expect(requestBody.tools).toEqual(expect.arrayContaining([expect.objectContaining({ name: 'compose_card' })]))
      expect(requestBody.instructions).toContain('Card Builder mode is active')
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({ output: [functionCall('compose_card', {
        node_id: 'agent-silu', label: null, category: 'activation', need: 'Apply a SiLU activation to hidden states',
        in_features: null, out_features: null, probability: null, input_role: null, output_role: null, reason: 'No existing card matched.',
      }, 'call-compose')] }), { status: 200 })
      const output = requestBody.input.find((item) => item.type === 'function_call_output')?.output ?? ''
      expect(output).toContain('nn.SiLU()')
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Composed the missing card', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({ request: 'Create the missing SiLU card', context: { cardBuilderMode: true, graph: { nodes: [] }, availableAtomics: [] } })
    expect(result.createdBlocks).toEqual([expect.objectContaining({ nodeId: 'agent-silu', pytorchModule: 'nn.SiLU()', inputRole: 'hidden', outputRole: 'hidden' })])
    expect(result.toolTrace).toEqual(expect.arrayContaining([expect.objectContaining({ tool: 'compose_card', status: 'accepted' })]))
  })

  it('keeps existing work read-only in parallel architecture mode', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestBody = JSON.parse(String(init?.body)) as { instructions: string }
      expect(requestBody.instructions).toContain('Operation mode is parallel architecture')
      expect(requestBody.instructions).toContain('existing node and connection as read-only')
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Parallel model', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    await askLabo({ request: 'Add another model', context: { operationMode: 'parallel', graph: { nodes: [] } } })
    expect(fetchMock).toHaveBeenCalledOnce()
  })

  it('keeps Edit Cards on the active selection and rejects Add Blocks tools', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestBody = JSON.parse(String(init?.body)) as { instructions: string; input: Array<{ type?: string; output?: string }> }
      expect(requestBody.instructions).toContain('Never add, clone, extract')
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({ output: [functionCall('add_block', { atom_id: 'relu', node_id: 'new-relu', reason: 'Wrong mode' }, 'call-add')] }), { status: 200 })
      if (fetchMock.mock.calls.length === 2) {
        expect(requestBody.input.find((item) => item.type === 'function_call_output' && item.output?.includes('Add Blocks is disabled'))).toBeTruthy()
        return new Response(JSON.stringify({ output: [functionCall('edit_card', { node_id: 'norm', label: 'Selected norm', settings_json: null, pytorch_module: null, reason: 'Edit selected card' }, 'call-edit')] }), { status: 200 })
      }
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Selected edit ready', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({
      request: 'Rename this selected card',
      context: {
        editing: { active: true, nodeIds: ['norm'] },
        graph: { nodes: [{ id: 'norm', atomId: 'rms-norm', label: 'RMSNorm', inputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }] }], connections: [] },
        availableAtomics: [{ atomId: 'relu', label: 'ReLU', inputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }] }],
      },
    })

    expect(result.addedBlocks).toHaveLength(0)
    expect(result.updatedBlocks).toEqual([expect.objectContaining({ nodeId: 'norm', label: 'Selected norm' })])
    expect(result.toolTrace).toEqual(expect.arrayContaining([
      expect.objectContaining({ tool: 'add_block', status: 'rejected' }),
      expect.objectContaining({ tool: 'edit_card', status: 'accepted' }),
    ]))
  })

  it('lets Add Blocks replace an existing card when the construction requires it', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async () => fetchMock.mock.calls.length === 1
      ? new Response(JSON.stringify({ output: [functionCall('replace_card', { node_id: 'activation', atom_id: 'gelu', reason: 'The requested architecture uses GELU instead of ReLU' }, 'call-replace')] }), { status: 200 })
      : new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Replacement ready', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({
      request: 'Build this architecture with GELU instead of the current ReLU',
      context: {
        graph: { nodes: [{ id: 'activation', atomId: 'relu', label: 'ReLU', inputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }] }], connections: [] },
        availableAtomics: [{ atomId: 'gelu', label: 'GELU', inputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }] }],
      },
    })

    expect(result.replacedBlocks).toEqual([{ nodeId: 'activation', atomId: 'gelu', reason: 'The requested architecture uses GELU instead of ReLU' }])
    expect(result.toolTrace).toEqual(expect.arrayContaining([expect.objectContaining({ tool: 'replace_card', status: 'accepted' })]))
  })

  it('finds GPT building blocks for a natural-language QA chatbot request', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestBody = JSON.parse(String(init?.body)) as { input: Array<{ type?: string; output?: string }> }
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({ output: [functionCall('search_cards', { query: 'simple QA chatbot', category: null }, 'call-search')] }), { status: 200 })
      const output = requestBody.input.find((item) => item.type === 'function_call_output')?.output ?? ''
      expect(output).toContain('causal-sdpa')
      expect(output).toContain('lm-head')
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'QA graph found', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    const availableAtomics = ['token-ids-input', 'token-embedding', 'qkv-projection', 'attention-head-layout', 'causal-sdpa', 'merge-attention-heads', 'attention-output-projection', 'rms-norm', 'residual-add', 'swiglu-mlp', 'lm-head', 'greedy-token-decoder'].map((atomId) => ({ atomId, label: atomId, inputs: [], outputs: [] }))

    await expect(askLabo({ request: 'Build a simple QA chatbot', context: { graph: { nodes: [] }, availableAtomics } })).resolves.toMatchObject({ summary: 'QA graph found', missingBlocks: [] })
  })

  it('deletes an architecture through one bulk agent tool', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async () => fetchMock.mock.calls.length === 1
      ? new Response(JSON.stringify({ output: [functionCall('delete_architecture', { architecture_id: 'architecture-1', reason: 'Clean comparison' }, 'call-delete')] }), { status: 200 })
      : new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Architecture cleaned', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({ request: 'Delete the first architecture', context: { graph: { nodes: [{ id: 'a', label: 'A' }, { id: 'b', label: 'B' }] }, architectures: [{ id: 'architecture-1', label: 'First', nodeIds: ['a', 'b'] }] } })
    expect(result.deletedBlocks).toEqual([{ nodeId: 'a', reason: 'Clean comparison (First)' }, { nodeId: 'b', reason: 'Clean comparison (First)' }])
    expect(result.toolTrace).toEqual(expect.arrayContaining([expect.objectContaining({ tool: 'delete_architecture', summary: expect.stringContaining('2 cards') })]))
  })

  it('discovers and wires every compatible typed port without guessing port ids', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async () => {
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({ output: [
        functionCall('add_block', { atom_id: 'head-layout', node_id: 'heads', reason: 'Prepare QKV heads' }, 'call-heads'),
        functionCall('add_block', { atom_id: 'causal-attention', node_id: 'attention', reason: 'Compute attention' }, 'call-attention'),
      ] }), { status: 200 })
      if (fetchMock.mock.calls.length === 2) return new Response(JSON.stringify({ output: [functionCall('connect_compatible', { source_id: 'heads', target_id: 'attention', connect_all: true, reason: 'Wire QKV safely' }, 'call-connect')] }), { status: 200 })
      if (fetchMock.mock.calls.length === 3) return new Response(JSON.stringify({ output: [functionCall('validate_graph', {}, 'call-validate')] }), { status: 200 })
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Typed attention graph ready', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({ request: 'Build typed attention', context: { graph: { nodes: [], connections: [] }, availableAtomics: [
      { atomId: 'head-layout', label: 'Head layout', inputs: [], outputs: [{ id: 'query', tensor: 'query', rank: 4 }, { id: 'key', tensor: 'key', rank: 4 }, { id: 'value', tensor: 'value', rank: 4 }] },
      { atomId: 'causal-attention', label: 'Causal attention', inputs: [{ id: 'query', tensor: 'query', rank: 4 }, { id: 'key', tensor: 'key', rank: 4 }, { id: 'value', tensor: 'value', rank: 4 }], outputs: [{ id: 'attention', tensor: 'attention', rank: 4 }] },
    ] } })

    expect(result.connections).toHaveLength(3)
    expect(result.connections.map((connection) => `${connection.sourcePortId}:${connection.targetPortId}`)).toEqual(['query:query', 'key:key', 'value:value'])
    expect(result.toolTrace).toEqual(expect.arrayContaining([
      expect.objectContaining({ tool: 'connect_compatible', status: 'accepted', summary: expect.stringContaining('3 compatible elastics') }),
      expect.objectContaining({ tool: 'validate_graph', status: 'read', summary: expect.stringContaining('Virtual graph valid') }),
    ]))
  })

  it('rejects an incomplete finish and lets the model repair the virtual graph', async () => {
    process.env.OPENAI_API_KEY = 'test-secret-key'
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestBody = JSON.parse(String(init?.body)) as { input: Array<{ type?: string; output?: string }> }
      if (fetchMock.mock.calls.length === 1) return new Response(JSON.stringify({ output: [
        functionCall('add_block', { atom_id: 'hidden-input', node_id: 'input', reason: 'Provide hidden states' }, 'call-input'),
        functionCall('add_block', { atom_id: 'relu', node_id: 'activation', reason: 'Apply activation' }, 'call-relu'),
      ] }), { status: 200 })
      if (fetchMock.mock.calls.length === 2) return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Premature plan', missing_blocks: [], warnings: [] }, 'call-premature')] }), { status: 200 })
      if (fetchMock.mock.calls.length === 3) {
        expect(requestBody.input.find((item) => item.type === 'function_call_output' && item.output?.includes('activation.hidden requires hidden'))).toBeTruthy()
        return new Response(JSON.stringify({ output: [functionCall('connect_compatible', { source_id: 'input', target_id: 'activation', connect_all: false, reason: 'Repair the missing input' }, 'call-repair')] }), { status: 200 })
      }
      return new Response(JSON.stringify({ output: [functionCall('finish_plan', { summary: 'Repaired plan', missing_blocks: [], warnings: [] }, 'call-finish')] }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await askLabo({ request: 'Build and repair an activation graph', context: { graph: { nodes: [], connections: [] }, availableAtomics: [
      { atomId: 'hidden-input', label: 'Hidden input', inputs: [], outputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }] },
      { atomId: 'relu', label: 'ReLU', inputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }] },
    ] } })

    expect(result.summary).toBe('Repaired plan')
    expect(result.connections).toEqual([expect.objectContaining({ sourceId: 'input', targetId: 'activation' })])
    expect(result.toolTrace).toEqual(expect.arrayContaining([
      expect.objectContaining({ tool: 'finish_plan', status: 'rejected', summary: expect.stringContaining('activation.hidden') }),
      expect.objectContaining({ tool: 'connect_compatible', status: 'accepted' }),
    ]))
  })
})
