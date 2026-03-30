import { describe, expect, it } from 'vitest'
import { connectCable, disconnectCable } from './cables'
import { addNode } from './ir'
import { gqaPreset, tokenMoePreset } from './presets'

describe('elastic typed cables', () => {
  it('allows one output to feed multiple compatible inputs', () => {
    const withoutHiddenFanout = {
      ...gqaPreset,
      edges: gqaPreset.edges.filter((edge) => !edge.id.startsWith('hidden-')),
    }
    const q = connectCable(withoutHiddenFanout, { sourceId: 'hidden', sourcePort: 'hidden', targetId: 'q-proj', targetPort: 'hidden' })
    const k = connectCable(q.graph, { sourceId: 'hidden', sourcePort: 'hidden', targetId: 'k-proj', targetPort: 'hidden' })
    const v = connectCable(k.graph, { sourceId: 'hidden', sourcePort: 'hidden', targetId: 'v-proj', targetPort: 'hidden' })

    expect(v.ok).toBe(true)
    expect(v.graph.edges.filter((edge) => edge.source === 'hidden')).toHaveLength(3)
  })

  it('replaces the cable on a single input while preserving other inputs', () => {
    const graph = addNode(gqaPreset, {
      id: 'q-alt', kind: 'linear', label: 'Alternative Q', role: 'query',
      position: { x: 10, y: 260 }, attributes: { inFeatures: 384, outFeatures: 384, bias: false },
    })
    const outcome = connectCable(graph, { sourceId: 'q-alt', sourcePort: 'query', targetId: 'sdpa', targetPort: 'query' })

    expect(outcome.ok).toBe(true)
    expect(outcome.graph.edges.filter((edge) => edge.target === 'sdpa')).toHaveLength(3)
    expect(outcome.graph.edges.some((edge) => edge.source === 'q-alt' && edge.target === 'sdpa')).toBe(true)
    expect(outcome.graph.edges.some((edge) => edge.source === 'q-proj' && edge.target === 'sdpa')).toBe(false)
  })

  it('rejects incompatible plugs without mutating the graph', () => {
    const outcome = connectCable(gqaPreset, { sourceId: 'v-proj', sourcePort: 'value', targetId: 'sdpa', targetPort: 'query' })
    expect(outcome).toMatchObject({ ok: false, code: 'PORT_TYPE_MISMATCH' })
    expect(outcome.graph).toBe(gqaPreset)
  })

  it('disconnects one cable without affecting the others', () => {
    const outcome = disconnectCable(gqaPreset, 'q-sdpa')
    expect(outcome.edges.some((edge) => edge.id === 'q-sdpa')).toBe(false)
    expect(outcome.edges.some((edge) => edge.id === 'k-sdpa')).toBe(true)
  })

  it('keeps distinct semantic input IDs even when their tensor types match', () => {
    const graph = { ...tokenMoePreset, edges: tokenMoePreset.edges.filter((edge) => edge.target !== 'merge') }
    const routed = connectCable(graph, {
      sourceId: 'norm', sourcePort: 'hidden', sourcePortId: 'output',
      targetId: 'merge', targetPort: 'hidden', targetPortId: 'routed',
    })
    const shared = connectCable(routed.graph, {
      sourceId: 'norm', sourcePort: 'hidden', sourcePortId: 'output',
      targetId: 'merge', targetPort: 'hidden', targetPortId: 'shared',
    })

    expect(shared.graph.edges.filter((edge) => edge.target === 'merge')).toEqual(expect.arrayContaining([
      expect.objectContaining({ targetPort: 'routed' }),
      expect.objectContaining({ targetPort: 'shared' }),
    ]))
  })
})
