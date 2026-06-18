import { describe, expect, it } from 'vitest'
import appSource from './App.tsx?raw'
import canvasSource from './model/GraphCanvas.tsx?raw'
import cableHookSource from './model/useElasticCables.ts?raw'
import irSource from './core/ir.ts?raw'

describe('renderer architecture boundaries', () => {
  it('contains no hardcoded lexical-GQA architecture or failed lexical lowering', () => {
    expect(irSource).not.toMatch(/lexical-gqa|recomposeWithLexicalKey|lexical_k_proj|lexical_gate/)
  })
  it('keeps App as a small studio shell instead of a graph-editor god component', () => {
    expect(appSource.split('\n').length).toBeLessThan(120)
    expect(appSource).not.toContain('compileToPyTorch')
    expect(appSource).not.toContain('ArchitectureNode')
    expect(appSource).not.toContain('connectCable')
    expect(appSource).toContain('<ModelStudio')
  })

  it('isolates canvas rendering and elastic cable interaction', () => {
    expect(canvasSource).toContain('GraphCanvas')
    expect(cableHookSource).toContain('useElasticCables')
    expect(cableHookSource).toContain('connectCable')
  })
})
