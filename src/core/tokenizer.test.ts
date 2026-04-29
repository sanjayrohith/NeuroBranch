import { describe, expect, it } from 'vitest'
import { createTokenizerArtifact, materializeToken } from './tokenizer'

describe('tokenizer artifacts', () => {
  it('derives vocabulary metadata and materializes selected token atoms', () => {
    const artifact = createTokenizerArtifact({
      id: 'fixture-bpe',
      name: 'Fixture BPE',
      family: 'bpe',
      checksum: 'sha256:test-only',
      tokens: [
        { id: 0, piece: '<eos>', bytes: [60, 101, 111, 115, 62], special: true, frequency: 12 },
        { id: 1, piece: ' attention', bytes: [32, 97, 116, 116, 101, 110, 116, 105, 111, 110], special: false, frequency: 2048 },
      ],
    })

    const token = materializeToken(artifact, 1)

    expect(artifact.vocabSize).toBe(2)
    expect(token.tokenizerId).toBe('fixture-bpe')
    expect(token.piece).toBe(' attention')
    expect(token.byteLength).toBe(10)
    expect(token.frequency).toBe(2048)
    expect(token.special).toBe(false)
  })
})
