import { describe, expect, it } from 'vitest'
import { cardsOverlap, resolveCardDrop, type CardPlacement } from './card-layout'

const card = (id: string, x: number, y: number): CardPlacement => ({ id, position: { x, y }, width: 120, height: 69 })

describe('model card layout', () => {
  it('moves a dropped card beyond every overlapping card in a dense column', () => {
    const others = [card('embedding', 190, 155), card('norm', 190, 245), card('qkv', 190, 355)]
    const resolved = resolveCardDrop({ id: 'dragged', original: { x: 190, y: 70 }, desired: { x: 190, y: 230 }, width: 120, height: 69 }, others)
    const dropped = card('dragged', resolved.x, resolved.y)

    expect(others.every((other) => !cardsOverlap(dropped, other, 16))).toBe(true)
    expect(resolved.y).toBeGreaterThan(355)
  })

  it('keeps the requested position when cards are in separate columns', () => {
    const desired = { x: 420, y: 245 }
    expect(resolveCardDrop({ id: 'dragged', original: { x: 190, y: 155 }, desired, width: 120, height: 69 }, [card('norm', 190, 245)])).toEqual(desired)
  })

  it('uses the opposite free slot instead of ejecting a card above the canvas', () => {
    const others = [card('tokens', 190, 75), card('embedding', 190, 155)]
    const resolved = resolveCardDrop({ id: 'norm', original: { x: 190, y: 245 }, desired: { x: 190, y: 155 }, width: 120, height: 69 }, others)
    const dropped = card('norm', resolved.x, resolved.y)

    expect(resolved.y).toBeGreaterThanOrEqual(50)
    expect(others.every((other) => !cardsOverlap(dropped, other, 16))).toBe(true)
  })
})
