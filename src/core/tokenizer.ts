export interface TokenRecord {
  id: number
  piece: string
  bytes: number[]
  special: boolean
  frequency?: number
}

export interface TokenizerArtifactInput {
  id: string
  name: string
  family: string
  checksum: string
  tokens: TokenRecord[]
}

export interface TokenizerArtifact extends TokenizerArtifactInput {
  vocabSize: number
  tokensById: ReadonlyMap<number, TokenRecord>
}

export interface MaterializedToken extends TokenRecord {
  tokenizerId: string
  byteLength: number
}

export function createTokenizerArtifact(input: TokenizerArtifactInput): TokenizerArtifact {
  const tokensById = new Map<number, TokenRecord>()
  for (const token of input.tokens) {
    if (tokensById.has(token.id)) throw new Error(`Duplicate token id: ${token.id}`)
    tokensById.set(token.id, Object.freeze({ ...token, bytes: [...token.bytes] }))
  }

  return Object.freeze({
    ...input,
    tokens: [...input.tokens],
    vocabSize: tokensById.size,
    tokensById,
  })
}

export function materializeToken(artifact: TokenizerArtifact, tokenId: number): MaterializedToken {
  const token = artifact.tokensById.get(tokenId)
  if (!token) throw new Error(`Token ${tokenId} does not exist in ${artifact.id}`)

  return {
    ...token,
    bytes: [...token.bytes],
    tokenizerId: artifact.id,
    byteLength: token.bytes.length,
  }
}
