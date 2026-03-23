import { describe, expect, it } from 'vitest'
import { modelAtomRegistry } from './model-atoms'

const requiredAtoms = [
  'token-embedding',
  'rms-norm',
  'layer-norm',
  'qkv-projection',
  'attention-head-layout',
  'rope',
  'causal-sdpa',
  'residual-add',
  'swiglu-mlp',
  'gelu-mlp',
  'lm-head',
  'cross-entropy-loss',
]

describe('full model atom registry', () => {
  it('offers at least 100 real cards without counting user-created cards', () => {
    expect(Object.keys(modelAtomRegistry).length).toBeGreaterThanOrEqual(100)
    expect(Object.values(modelAtomRegistry).filter((atom) => !atom.composite).length).toBeGreaterThanOrEqual(100)
  })

  it('defines typed, executable atoms for a complete GPT path', () => {
    for (const atomId of requiredAtoms) {
      const atom = modelAtomRegistry[atomId]
      expect(atom, `missing ${atomId}`).toBeDefined()
      expect(atom.inputs.length, `${atomId} inputs`).toBeGreaterThan(0)
      expect(atom.outputs.length, `${atomId} outputs`).toBeGreaterThan(0)
      expect(atom.lowerings.pytorch, `${atomId} PyTorch lowering`).toMatchObject({ executable: true })
      expect(atom.lowerings.pytorch.forward.length, `${atomId} forward`).toBeGreaterThan(0)
    }

    expect(modelAtomRegistry['attention-head-layout'].settings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'queryHeads' }),
        expect.objectContaining({ id: 'keyValueHeads' }),
        expect.objectContaining({ id: 'headDim' }),
      ]),
    )
    expect(modelAtomRegistry['lm-head'].settings).toContainEqual(
      expect.objectContaining({ id: 'tieEmbeddingWeights', type: 'boolean' }),
    )
  })

  it('gives every public atomic block a structured executable PyTorch contract', () => {
    for (const atom of Object.values(modelAtomRegistry)) {
      if (atom.composite) continue
      expect(atom.lowerings.pytorch, `${atom.id} structured lowering`).toMatchObject({ executable: true })
      expect(Array.isArray(atom.lowerings.pytorch.declarations), `${atom.id} declarations`).toBe(true)
      expect(atom.lowerings.pytorch.forward.length, `${atom.id} forward`).toBeGreaterThan(0)
      expect(atom.lowerings.pytorch.forward.join('\n'), `${atom.id} placeholder helper`).not.toMatch(/build_qkv|semantic_topk|deepseek_style_moe|\bsettings\b/)
    }
  })

  it('exposes semantic DeepSeek-style MoE pieces without kernel plumbing', () => {
    const required = [
      'moe-router',
      'top-k-routing',
      'routed-expert-bank',
      'shared-expert-bank',
      'expert-merge',
      'load-balancing-loss',
      'deepseek-moe',
    ]
    for (const id of required) expect(modelAtomRegistry[id], `missing ${id}`).toBeDefined()

    expect(modelAtomRegistry['moe-router'].outputs).toEqual([
      { id: 'scores', tensor: 'routing-logits', rank: 3 },
    ])
    expect(modelAtomRegistry['top-k-routing']).toMatchObject({
      inputs: [{ id: 'scores', tensor: 'routing-logits', rank: 3 }],
      outputs: [
        { id: 'expertIndices', tensor: 'expert-indices', rank: 3 },
        { id: 'expertWeights', tensor: 'routing-weights', rank: 3 },
      ],
    })
    expect(modelAtomRegistry['routed-expert-bank'].inputs.map((port) => port.tensor)).toEqual([
      'hidden', 'expert-indices', 'routing-weights',
    ])
    expect(modelAtomRegistry['deepseek-moe'].composite?.atomIds).toEqual([
      'moe-router', 'top-k-routing', 'routed-expert-bank', 'shared-expert-bank', 'expert-merge',
    ])

    const publicIds = Object.keys(modelAtomRegistry).join(' ')
    expect(publicIds).not.toMatch(/dispatch|gather|scatter|all-to-all/)
  })
})
