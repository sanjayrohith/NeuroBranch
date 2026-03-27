import { describe, expect, it } from 'vitest'
import { executionLayers } from './execution-plan'
import { complexityDeepPreset } from './presets'

describe('elastic execution plan', () => {
  it('groups independent branches into stable parallel levels', () => {
    expect(executionLayers(complexityDeepPreset)).toEqual([
      ['tokens'],
      ['embedding', 'fixed-routes'],
      ['attention-norm'],
      ['qkv'],
      ['head-layout'],
      ['qk-norm'],
      ['rope'],
      ['kv-expand'],
      ['sdpa'],
      ['merge-heads'],
      ['attention-output'],
      ['attention-residual'],
      ['mlp-norm'],
      ['shared', 'routed'],
      ['branch-gates'],
      ['block-residual'],
      ['final-norm'],
      ['head'],
    ])
  })

  it('plays two disconnected architectures in parallel by execution depth', () => {
    const graph = {
      ...complexityDeepPreset,
      nodes: [
        { id: 'a-input', kind: 'input' as const, label: 'A input', role: 'hidden' as const, position: { x: 0, y: 0 } },
        { id: 'b-input', kind: 'input' as const, label: 'B input', role: 'hidden' as const, position: { x: 0, y: 0 } },
        { id: 'a-output', kind: 'semantic' as const, atomId: 'identity', label: 'A output', role: 'hidden' as const, position: { x: 0, y: 0 } },
        { id: 'b-output', kind: 'semantic' as const, atomId: 'identity', label: 'B output', role: 'hidden' as const, position: { x: 0, y: 0 } },
      ],
      edges: [
        { id: 'a', source: 'a-input', target: 'a-output' },
        { id: 'b', source: 'b-input', target: 'b-output' },
      ],
    }
    expect(executionLayers(graph)).toEqual([
      ['a-input', 'b-input'],
      ['a-output', 'b-output'],
    ])
  })
})
