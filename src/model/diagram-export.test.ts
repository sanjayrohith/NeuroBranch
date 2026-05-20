import { describe, expect, it } from 'vitest'
import { gptLikeStarterPreset } from '../core/presets'
import { architectureDiagramSvg, exportFileName } from './diagram-export'

describe('architecture diagram export', () => {
  it('renders a standalone vector graph with cards and cables', () => {
    const svg = architectureDiagramSvg(gptLikeStarterPreset)

    expect(svg).toMatch(/^<svg/)
    expect(svg).toContain('<path d="M ')
    expect(svg).toContain('Tied token embedding')
    expect(svg).toContain(`${gptLikeStarterPreset.nodes.length} cards`)
    expect(svg).toContain(`${gptLikeStarterPreset.edges.length} links`)
    expect(svg).not.toContain('[object Object]')
  })

  it('creates a filesystem-safe descriptive filename', () => {
    expect(exportFileName({ ...gptLikeStarterPreset, name: 'GPT / Démo 300M' }, 'svg')).toBe('gpt-d-mo-300m.svg')
  })

  it('preserves reviewed positions and separates parallel port anchors', () => {
    const graph = {
      ...gptLikeStarterPreset,
      nodes: gptLikeStarterPreset.nodes.slice(0, 2).map((node, index) => ({ ...node, position: { x: 100 + index * 310, y: 120 } })),
      edges: [
        { id: 'q', source: gptLikeStarterPreset.nodes[0]!.id, sourcePort: 'query', target: gptLikeStarterPreset.nodes[1]!.id, targetPort: 'query', label: 'Q' },
        { id: 'k', source: gptLikeStarterPreset.nodes[0]!.id, sourcePort: 'key', target: gptLikeStarterPreset.nodes[1]!.id, targetPort: 'key', label: 'K' },
      ],
    }
    const svg = architectureDiagramSvg(graph)

    const paths = [...svg.matchAll(/<path d="([^"]+)" fill="none" stroke=/g)].map((match) => match[1]).filter((path) => path.includes(' C '))
    expect(paths).toHaveLength(2)
    expect(paths[0]).not.toBe(paths[1])
    expect(svg).toContain('<title>query</title>')
    expect(svg).toContain('<title>key</title>')
  })
})
