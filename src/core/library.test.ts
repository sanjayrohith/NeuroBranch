import { describe, expect, it } from 'vitest'
import { attentionAtoms } from './library'

describe('QKV ablation library', () => {
  it('keeps executable lowering, tensor contracts, and provenance together', () => {
    const qkNorm = attentionAtoms.find((atom) => atom.id === 'qk-norm')
    const groupedQuery = attentionAtoms.find((atom) => atom.id === 'grouped-query')

    expect(qkNorm?.category).toBe('normalization')
    expect(qkNorm?.pytorch).toContain('rms_norm')
    expect(qkNorm?.provenance.title).toBeTruthy()
    expect(qkNorm?.contracts.preservesSequenceLength).toBe(true)
    expect(groupedQuery?.category).toBe('head-topology')
    expect(groupedQuery?.provenance.year).toBeGreaterThanOrEqual(2019)
  })
})
