import { optimizerRegistry, type OptimizerDefinition } from './core/optimizer-ir'
import { tokenizerAtomDefinitions, type TokenizerAtomKind } from './core/tokenizer-ir'
import type { CustomTokenizerCard } from './tokenizer/custom-tokenizer-card'

export type StudioSearchResult =
  | { id: string; kind: 'optimizer'; label: string; description: string }
  | { id: TokenizerAtomKind; kind: 'tokenizer-atom'; label: string; description: string }
  | { id: string; kind: 'tokenizer-custom'; label: string; description: string }

function normalize(value: string): string {
  return value.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim()
}

function rank<T extends { id: string; label: string; description: string }>(query: string, values: T[]): T[] {
  const normalized = normalize(query)
  if (!normalized) return []
  const terms = normalized.split(' ').filter(Boolean)
  return values.map((value) => {
    const haystack = normalize(`${value.id} ${value.label} ${value.description}`)
    const score = (haystack.includes(normalized) ? 20 : 0) + terms.reduce((total, term) => total + (haystack.includes(term) ? 3 : 0), 0)
    return { value, score }
  }).filter(({ score }) => score > 0).sort((left, right) => right.score - left.score || left.value.label.localeCompare(right.value.label)).slice(0, 12).map(({ value }) => value)
}

export function searchOptimizers(query: string, custom: OptimizerDefinition[]): StudioSearchResult[] {
  return rank(query, [...Object.values(optimizerRegistry), ...custom].map((optimizer) => ({
    id: optimizer.id,
    kind: 'optimizer' as const,
    label: optimizer.label,
    description: `${optimizer.composition ? optimizer.torchClass : `torch.optim.${optimizer.torchClass}`}${optimizer.notes ? ` · ${optimizer.notes}` : ''}`,
  })))
}

export function searchTokenizerCards(query: string, custom: CustomTokenizerCard[]): StudioSearchResult[] {
  const builtIn = (Object.entries(tokenizerAtomDefinitions) as Array<[TokenizerAtomKind, (typeof tokenizerAtomDefinitions)[TokenizerAtomKind]]>)
    .filter(([atom]) => atom !== 'custom-tokenizer')
    .map(([atom, definition]) => ({ id: atom, kind: 'tokenizer-atom' as const, label: definition.label, description: definition.category }))
  const userCards = custom.map((card) => ({ id: card.id, kind: 'tokenizer-custom' as const, label: card.label, description: `${card.category} · My cards` }))
  return rank(query, [...builtIn, ...userCards])
}
