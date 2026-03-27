import { describe, expect, it } from 'vitest'
import { architectureComponents } from './graph-components'
import { gptLikeStarterPreset } from './presets'

describe('architecture components', () => {
  it('uses known preset names and stable incremental names for unknown parallel graphs', () => {
    const graph = {
      ...gptLikeStarterPreset,
      nodes: [...gptLikeStarterPreset.nodes, { id: 'other-input', kind: 'input' as const, label: 'Other input', role: 'hidden' as const, position: { x: 900, y: 10 } }],
    }
    const components = architectureComponents(graph, [gptLikeStarterPreset])
    expect(components.map((component) => component.label)).toEqual(['GPT-like Starter', 'Architecture 2 · Other input'])
    expect(components[0]?.graph.nodes).toHaveLength(gptLikeStarterPreset.nodes.length)
    expect(components[1]?.nodeIds).toEqual(['other-input'])
  })
})
