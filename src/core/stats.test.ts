import { describe, expect, it } from 'vitest'
import { deriveGraphStats } from './stats'
import { gqaPreset } from './presets'

describe('derived graph statistics', () => {
  it('computes parameter counts from atomic node attributes', () => {
    const stats = deriveGraphStats(gqaPreset)

    expect(stats.parameterCount).toBe(368_640)
    expect(stats.nodeCount).toBe(gqaPreset.nodes.length)
    expect(stats.edgeCount).toBe(gqaPreset.edges.length)
    expect(stats.formattedParameterCount).toBe('368.64K')
  })
})
