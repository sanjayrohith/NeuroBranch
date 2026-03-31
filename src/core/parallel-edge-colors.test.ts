import { describe, expect, it } from 'vitest'
import { parallelEdgeColors } from './parallel-edge-colors'
import { blankStarterPreset } from './presets'

describe('parallel elastic colors', () => {
  it('keeps a stable pastel identity from a fan-out until its merge', () => {
    const graph = {
      ...blankStarterPreset,
      nodes: ['source', 'left', 'left-deep', 'right', 'merge'].map((id) => ({
        id,
        kind: 'semantic' as const,
        atomId: 'identity',
        label: id,
        role: 'hidden' as const,
        position: { x: 0, y: 0 },
      })),
      edges: [
        { id: 'source-left', source: 'source', target: 'left' },
        { id: 'left-deep', source: 'left', target: 'left-deep' },
        { id: 'left-merge', source: 'left-deep', target: 'merge' },
        { id: 'source-right', source: 'source', target: 'right' },
        { id: 'right-merge', source: 'right', target: 'merge' },
      ],
    }
    const colors = parallelEdgeColors(graph)

    expect(colors.get('source-left')).toBe(colors.get('left-deep'))
    expect(colors.get('left-deep')).toBe(colors.get('left-merge'))
    expect(colors.get('source-right')).toBe(colors.get('right-merge'))
    expect(colors.get('source-left')).not.toBe(colors.get('source-right'))
  })
})
