import { describe, expect, it } from 'vitest'
import { blankStarterPreset } from '../core/presets'
import { orderedNodeInputPorts } from './port-layout'

describe('semantic input plug layout', () => {
  it('visually aligns equivalent H/H ports with their source order to avoid crossed elastics', () => {
    const graph = {
      ...blankStarterPreset,
      nodes: [
        { id: 'left', kind: 'semantic' as const, atomId: 'identity', label: 'Left', role: 'hidden' as const, position: { x: 100, y: 100 } },
        { id: 'right', kind: 'semantic' as const, atomId: 'identity', label: 'Right', role: 'hidden' as const, position: { x: 500, y: 100 } },
        { id: 'merge', kind: 'semantic' as const, atomId: 'residual-add', label: 'Merge', role: 'hidden' as const, position: { x: 300, y: 300 } },
      ],
      edges: [
        { id: 'right-residual', source: 'right', sourcePort: 'output', target: 'merge', targetPort: 'residual' },
        { id: 'left-branch', source: 'left', sourcePort: 'output', target: 'merge', targetPort: 'branch' },
      ],
    }

    expect(orderedNodeInputPorts(graph, graph.nodes[2]!).map((port) => port.id)).toEqual(['branch', 'residual'])
  })

  it('does not reorder ports with different tensor types', () => {
    const graph = {
      ...blankStarterPreset,
      nodes: [{ id: 'topk', kind: 'semantic' as const, atomId: 'top-k-routing', label: 'Top K', role: 'hidden' as const, position: { x: 0, y: 0 } }],
      edges: [],
    }
    expect(orderedNodeInputPorts(graph, graph.nodes[0]!).map((port) => port.id)).toEqual(['scores'])
  })
})
