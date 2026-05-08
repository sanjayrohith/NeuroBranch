export interface CardPlacement {
  id: string
  position: { x: number; y: number }
  width: number
  height: number
}

export interface CardDrop {
  id: string
  original: { x: number; y: number }
  desired: { x: number; y: number }
  width: number
  height: number
}

export const MODEL_CARD_WIDTH = 148
export const MODEL_CARD_HEIGHT = 76
export const MODEL_CARD_GAP = 16

export function cardsOverlap(left: CardPlacement, right: CardPlacement, gap = 0): boolean {
  return Math.abs(left.position.x - right.position.x) < (left.width + right.width) / 2 + gap
    && Math.abs(left.position.y - right.position.y) < (left.height + right.height) / 2 + gap
}

export function resolveCardDrop(drop: CardDrop, others: CardPlacement[], gap = MODEL_CARD_GAP): { x: number; y: number } {
  const preferredDirection = drop.desired.y >= drop.original.y ? 1 : -1

  const resolveInDirection = (direction: number) => {
    const resolved = { ...drop.desired }

    for (let attempt = 0; attempt <= others.length; attempt += 1) {
      const current: CardPlacement = { id: drop.id, position: resolved, width: drop.width, height: drop.height }
      const collisions = others.filter((other) => cardsOverlap(current, other, gap))
      if (collisions.length === 0) return resolved

      if (direction > 0) {
        resolved.y = Math.max(...collisions.map((other) => other.position.y + (drop.height + other.height) / 2 + gap))
      } else {
        resolved.y = Math.min(...collisions.map((other) => other.position.y - (drop.height + other.height) / 2 - gap))
      }
    }

    return resolved
  }

  const preferred = resolveInDirection(preferredDirection)
  const minimumY = drop.height / 2 + gap
  return preferred.y >= minimumY ? preferred : resolveInDirection(-preferredDirection)
}
