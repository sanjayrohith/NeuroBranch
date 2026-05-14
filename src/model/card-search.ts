import { modelAtomRegistry } from '../core/model-atoms'

export interface ModelCardSearchResult {
  atomId: string
  label: string
  description: string
  kind: 'atomic' | 'input'
}

const inputCards: ModelCardSearchResult[] = [
  { atomId: 'token-ids-input', label: 'Token IDs', description: 'Source de tokens pour embedding et génération', kind: 'input' },
  { atomId: 'hidden-state-input', label: 'Hidden State', description: 'Entrée directe de tenseurs cachés', kind: 'input' },
  { atomId: 'training-labels-input', label: 'Training Labels', description: 'Labels Y pour les pertes d’entraînement', kind: 'input' },
]

const reusableCardInputs: ModelCardSearchResult[] = [
  ...inputCards,
  { atomId: 'image-input', label: 'Image tensor', description: 'Image input exposed by the reusable card', kind: 'input' },
  { atomId: 'video-input', label: 'Video tensor', description: 'Video input exposed by the reusable card', kind: 'input' },
  { atomId: 'audio-input', label: 'Audio tensor', description: 'Audio input exposed by the reusable card', kind: 'input' },
]

const intentAliases: Record<string, string[]> = {
  generate: ['generation', 'generer', 'generated', 'autoregressif', 'autoregressive', 'inference', 'decodeur', 'decoder', 'sampler', 'echantillonneur', 'token'],
  sampler: ['sampling', 'echantillon', 'decodeur', 'decoder', 'generation', 'logits', 'token'],
  normalization: ['normaliser', 'normalisation', 'norme', 'norm', 'stabiliser'],
  attention: ['attendre', 'contexte', 'contextuel', 'query', 'key', 'value', 'qkv', 'gqa', 'mha'],
  routing: ['routeur', 'routage', 'expert', 'experts', 'moe', 'mixture'],
  embedding: ['plongement', 'embed', 'vocabulaire', 'vocabulary', 'token'],
  objective: ['loss', 'perte', 'entrainement', 'training', 'label', 'labels'],
  activation: ['relu', 'gelu', 'silu', 'swish', 'activation', 'nonlineaire'],
  position: ['position', 'positionnel', 'rope', 'rotary', 'sinusoidal'],
  output: ['sortie', 'logits', 'prediction', 'predire', 'vocabulaire'],
  media: ['image', 'vision', 'video', 'audio', 'speech', 'waveform', 'frame', 'multimodal', 'patch', 'latent', 'diffusion', 'denoiser', 'conditionnement', 'conditioning'],
}

function normalize(value: string): string {
  return value.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim()
}

function expandedTerms(query: string): string[] {
  const terms = new Set(normalize(query).split(' ').filter((term) => term.length > 1))
  for (const [intent, aliases] of Object.entries(intentAliases)) {
    if (terms.has(intent) || aliases.some((alias) => terms.has(normalize(alias)))) {
      terms.add(intent)
      aliases.forEach((alias) => terms.add(normalize(alias)))
    }
  }
  return [...terms]
}

function searchCards(query: string, cards: ModelCardSearchResult[], limit: number): ModelCardSearchResult[] {
  const normalizedQuery = normalize(query)
  if (!normalizedQuery) return []
  const terms = expandedTerms(query)
  const atomCards: ModelCardSearchResult[] = Object.values(modelAtomRegistry)
    .filter((definition) => !definition.composite)
    .map((definition) => ({ atomId: definition.id, label: definition.label, description: `${definition.category} · ${definition.inputs.map((port) => port.tensor).join(' + ') || 'source'} → ${definition.outputs.map((port) => port.tensor).join(' + ')}`, kind: 'atomic' }))

  return [...cards, ...atomCards]
    .map((card) => {
      const haystack = normalize(`${card.atomId} ${card.label} ${card.description}`)
      let score = haystack.includes(normalizedQuery) ? 20 : 0
      for (const term of terms) {
        if (haystack.split(' ').includes(term)) score += 5
        else if (haystack.includes(term)) score += 2
      }
      if (/logits?.*(token|gener|sampl|decod)|(?:token|gener|sampl|decod).*logits?/.test(normalizedQuery)
        && ['greedy-token-decoder', 'top-k-token-sampler', 'multinomial-token-sampler'].includes(card.atomId)) score += 40
      if (/source.*token|token.*(?:source|entree|input|ids)/.test(normalizedQuery) && card.atomId === 'token-ids-input') score += 40
      return { card, score }
    })
    .filter(({ score }) => score > 0)
    .sort((left, right) => right.score - left.score || left.card.label.localeCompare(right.card.label))
    .slice(0, limit)
    .map(({ card }) => card)
}

export function searchModelCards(query: string, limit = 10): ModelCardSearchResult[] {
  return searchCards(query, inputCards, limit)
}

export function searchReusableCardAtoms(query: string, limit = 10): ModelCardSearchResult[] {
  return searchCards(query, reusableCardInputs, limit)
}
