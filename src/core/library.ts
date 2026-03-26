export type AtomCategory =
  | 'projection'
  | 'head-topology'
  | 'normalization'
  | 'position'
  | 'score'
  | 'routing'

export interface AttentionAtom {
  id: string
  label: string
  category: AtomCategory
  pytorch: string
  contracts: {
    preservesSequenceLength: boolean
    causalSafe: boolean
  }
  provenance: {
    title: string
    year: number
    url: string
  }
}

export const attentionAtoms: AttentionAtom[] = [
  {
    id: 'grouped-query',
    label: 'Grouped-query heads',
    category: 'head-topology',
    pytorch: 'k = k.repeat_interleave(query_heads // key_value_heads, dim=1)',
    contracts: { preservesSequenceLength: true, causalSafe: true },
    provenance: {
      title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
      year: 2023,
      url: 'https://arxiv.org/abs/2305.13245',
    },
  },
  {
    id: 'qk-norm',
    label: 'Query-key normalization',
    category: 'normalization',
    pytorch: 'q = rms_norm(q); k = rms_norm(k)',
    contracts: { preservesSequenceLength: true, causalSafe: true },
    provenance: {
      title: 'Query-Key Normalization for Transformers',
      year: 2020,
      url: 'https://arxiv.org/abs/2010.04245',
    },
  },
  {
    id: 'causal-sdpa',
    label: 'Causal scaled dot-product attention',
    category: 'score',
    pytorch: 'F.scaled_dot_product_attention(q, k, v, is_causal=True)',
    contracts: { preservesSequenceLength: true, causalSafe: true },
    provenance: {
      title: 'Attention Is All You Need',
      year: 2017,
      url: 'https://arxiv.org/abs/1706.03762',
    },
  },
]
