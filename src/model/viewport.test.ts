import { describe, expect, it } from 'vitest'
import { clampZoom, graphGridStyle, panViewport, screenToWorld, zoomViewportAt } from './viewport'

describe('2D graph viewport geometry', () => {
  it('zooms around the pointer without moving the world point under it', () => {
    const initial = { x: -120, y: 80, zoom: 1 }
    const pointer = { x: 420, y: 260 }
    const worldBefore = screenToWorld(pointer, initial)
    const zoomed = zoomViewportAt(initial, 1.75, pointer)
    expect(screenToWorld(pointer, zoomed)).toEqual(worldBefore)
  })

  it('pans independently on both axes', () => {
    expect(panViewport({ x: 10, y: 20, zoom: 1 }, -35, 48)).toEqual({ x: -25, y: 68, zoom: 1 })
  })

  it('clamps zoom to a usable node-editor range', () => {
    expect(clampZoom(0.01)).toBe(0.2)
    expect(clampZoom(3)).toBe(2.4)
    expect(clampZoom(1.25)).toBe(1.25)
  })

  it('converts screen coordinates back into graph-world coordinates', () => {
    expect(screenToWorld({ x: 300, y: 180 }, { x: 100, y: -20, zoom: 2 })).toEqual({ x: 100, y: 100 })
  })

  it('keeps the repeating grid aligned with arbitrarily distant pan and zoom values', () => {
    expect(graphGridStyle({ x: -10_003, y: 20_011, zoom: 1.5 })).toEqual({
      backgroundPosition: '5px 31px',
      backgroundSize: '36px 36px',
    })
  })
})
