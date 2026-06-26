import { spawnSync } from 'node:child_process'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { removeNode } from '../src/core/ir'
import { complexityDeepPreset, gptLikeStarterPreset, gqaPreset, tokenMoePreset } from '../src/core/presets'
import type { ArchitectureGraph } from '../src/core/ir'
import { researchBpePreset } from '../src/core/tokenizer-presets'

const root = process.cwd()
const python = join(root, '.venv', 'bin', 'python')
const runner = join(root, 'scripts', 'atomic_runtime.py')

const atomicComplexityGraph = {
  ...structuredClone(complexityDeepPreset),
  config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 1, headDim: 8 },
  nodes: complexityDeepPreset.nodes.map((node) => {
    if (node.id === 'embedding') return { ...node, attributes: { vocabSize: 64, hiddenSize: 16 } }
    if (node.id === 'head-layout') return { ...node, attributes: { queryHeads: 2, keyValueHeads: 1, headDim: 8 } }
    if (node.id === 'fixed-routes') return { ...node, attributes: { vocabSize: 64, nExperts: 4, topK: 2, layerIndex: 0, primaryWeight: 0.5 } }
    if (node.id === 'shared') return { ...node, attributes: { nSharedExperts: 1, intermediateSize: 32, activation: 'swiglu' } }
    if (node.id === 'routed') return { ...node, attributes: { nExperts: 4, intermediateSize: 8, activation: 'swiglu', expertParallelSize: 1 } }
    if (node.id === 'head') return { ...node, attributes: { vocabSize: 64, tieEmbeddingWeights: true, bias: false } }
    return node
  }),
} as ArchitectureGraph

function run(payload: unknown) {
  const result = spawnSync(python, [runner], {
    cwd: root,
    input: JSON.stringify(payload),
    encoding: 'utf8',
  })
  expect(result.status, result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

describe('atomic Python runtime', () => {
  it('executes all semantic paper blocks in elastic dataflow order', () => {
    const trace = run({ kind: 'model', graph: atomicComplexityGraph })

    expect(trace.status).toBe('completed')
    expect(trace.results).toHaveLength(20)
    expect(trace.results.every((item: { status: string }) => item.status === 'passed')).toBe(true)
  })

  it('records the first failure while resolving every independent atom', () => {
    const graph = {
      ...atomicComplexityGraph,
      edges: atomicComplexityGraph.edges.filter((edge) => edge.id !== 'attention-norm-qkv'),
    }
    const trace = run({ kind: 'model', graph })

    expect(trace.status).toBe('failed')
    expect(trace.currentAtomId).toBe('qkv')
    expect(trace.results).toHaveLength(atomicComplexityGraph.nodes.length)
    expect(trace.results.find((item: { atomId: string }) => item.atomId === 'qkv')).toMatchObject({ status: 'failed' })
    expect(trace.results.find((item: { atomId: string }) => item.atomId === 'fixed-routes')).toMatchObject({ status: 'passed' })
  })

  it('executes the model graph atom by atom with real PyTorch tensors', () => {
    const trace = run({ kind: 'model', graph: gqaPreset })

    expect(trace.engine).toBe('pytorch')
    expect(trace.status).toBe('completed')
    expect(trace.results.map((item: { atomId: string }) => item.atomId)).toEqual([
      'hidden', 'q-proj', 'k-proj', 'v-proj', 'sdpa', 'output',
    ])
    expect(trace.results.every((item: { status: string }) => item.status === 'passed')).toBe(true)
    expect(trace.results[1].summary).toContain('[2, 8, 384]')
  })

  it('reports the exact root failure and marks its downstream as blocked', () => {
    const trace = run({ kind: 'model', graph: removeNode(gqaPreset, 'q-proj') })

    expect(trace.status).toBe('failed')
    expect(trace.currentAtomId).toBe('sdpa')
    expect(trace.results.find((item: { atomId: string }) => item.atomId === 'sdpa')).toMatchObject({ status: 'failed' })
    expect(trace.results.find((item: { atomId: string }) => item.atomId === 'output')).toMatchObject({ status: 'failed', error: expect.stringContaining('blocked by failed dependency') })
  })

  it('executes tokenizer atoms with the real tokenizers backend', () => {
    const trace = run({ kind: 'tokenizer', pipeline: researchBpePreset, sample: 'Café LABO AI' })

    expect(trace.engine).toBe('tokenizers')
    expect(trace.status).toBe('completed')
    expect(trace.results).toHaveLength(researchBpePreset.steps.length)
    expect(trace.results.every((item: { status: string }) => item.status === 'passed')).toBe(true)
    expect(trace.tokenIds).toEqual(expect.arrayContaining([expect.any(Number)]))
  })

  it('injects tokenizer-produced IDs into the model token input', () => {
    const tokenTrace = run({ kind: 'tokenizer', pipeline: researchBpePreset, sample: 'Bonjour LABO AI' })
    const modelTrace = run({ kind: 'model', graph: gptLikeStarterPreset, tokenIds: tokenTrace.tokenIds })

    expect(modelTrace.status).toBe('completed')
    expect(modelTrace.results[0].summary).toContain(`shape=[1, ${tokenTrace.tokenIds.length}]`)
    expect(modelTrace.results.at(-1)).toMatchObject({ atomId: 'head', status: 'passed' })
    expect(modelTrace.modelOutput).toMatchObject({
      kind: 'logits',
      tensorShape: [1, tokenTrace.tokenIds.length, 32000],
      logitsShape: [1, tokenTrace.tokenIds.length, 32000],
      predictedTokenId: expect.any(Number),
    })
    expect(modelTrace.modelOutput.topTokenIds).toHaveLength(5)
    expect(modelTrace.modelOutput.topProbabilities).toHaveLength(5)
  })

  it('runs learned routing, top-k experts and merge with real tensors', () => {
    const trace = run({ kind: 'model', graph: tokenMoePreset, tokenIds: [1, 2, 3] })

    expect(trace.status).toBe('completed')
    expect(trace.results.map((result: { atomId: string; status: string }) => [result.atomId, result.status])).toEqual([
      ['tokens', 'passed'], ['embedding', 'passed'], ['norm', 'passed'], ['router', 'passed'], ['topk', 'passed'],
      ['routed', 'passed'], ['shared', 'passed'], ['merge', 'passed'], ['head', 'passed'],
    ])
    expect(trace.modelOutput).toMatchObject({ kind: 'logits', tensorShape: [1, 3, 32768] })
  })
})
