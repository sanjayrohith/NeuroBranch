import { describe, expect, it } from 'vitest'
import { searchModelCards } from './card-search'

describe('natural-language model card search', () => {
  it('finds native logits decoders from a French generation request', () => {
    expect(searchModelCards('je veux convertir mes logits en token généré')[0]?.atomId).toBe('greedy-token-decoder')
  })

  it('finds the virtual Token IDs source instead of reporting it missing', () => {
    expect(searchModelCards('une source entrée token ids pour mon embedding')[0]?.atomId).toBe('token-ids-input')
  })

  it('understands semantic category language', () => {
    expect(searchModelCards('normaliser les hidden states').some((card) => card.atomId === 'rms-norm')).toBe(true)
  })
})
