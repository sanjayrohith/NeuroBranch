import { describe, expect, it } from 'vitest'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { resolveAtomicRuntimePaths, runAtomicRuntime } from './atomic-runtime'
import { gptLikeStarterPreset, gqaPreset, tokenMoePreset, trBasicPreset } from '../src/core/presets'
import { activationAtomRegistry } from '../src/core/activation-atoms'
import type { ArchitectureGraph } from '../src/core/ir'
import { audioEncoderPreset, multimodalImageEditorPreset, videoTransformerPreset, visionTransformerPreset } from '../src/core/media-presets'

describe('Electron atomic runtime bridge', () => {
  it('resolves the development venv and runner outside Electron archives', () => {
    expect(resolveAtomicRuntimePaths({
      projectRoot: process.cwd(),
      resourcesPath: '/missing-resources',
      homeDirectory: '/missing-home',
    })).toEqual({
      pythonExecutable: `${process.cwd()}/.venv/bin/python`,
      runnerScript: `${process.cwd()}/scripts/atomic_runtime.py`,
    })
  })

  it('resolves the standard Windows virtual-environment layout', () => {
    const root = mkdtempSync(join(tmpdir(), 'labo-ai-windows-'))
    try {
      mkdirSync(join(root, '.venv', 'Scripts'), { recursive: true })
      mkdirSync(join(root, 'scripts'), { recursive: true })
      writeFileSync(join(root, '.venv', 'Scripts', 'python.exe'), '')
      writeFileSync(join(root, 'scripts', 'atomic_runtime.py'), '')

      expect(resolveAtomicRuntimePaths({ projectRoot: root, resourcesPath: join(root, 'resources'), homeDirectory: join(root, 'home'), environmentPath: '', platform: 'win32' })).toEqual({
        pythonExecutable: join(root, '.venv', 'Scripts', 'python.exe'),
        runnerScript: join(root, 'scripts', 'atomic_runtime.py'),
      })
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('prefers the managed installed runtime over PATH Python', () => {
    const root = mkdtempSync(join(tmpdir(), 'labo-ai-managed-runtime-'))
    try {
      const userData = join(root, 'user-data')
      const resources = join(root, 'resources')
      mkdirSync(join(userData, 'runtime', 'Scripts'), { recursive: true })
      mkdirSync(join(resources, 'runtime'), { recursive: true })
      writeFileSync(join(userData, 'runtime', 'Scripts', 'python.exe'), '')
      writeFileSync(join(resources, 'runtime', 'atomic_runtime.py'), '')

      expect(resolveAtomicRuntimePaths({
        projectRoot: join(root, 'missing-project'),
        resourcesPath: resources,
        homeDirectory: join(root, 'missing-home'),
        userDataDirectory: userData,
        environmentPath: '',
        platform: 'win32',
      })).toEqual({
        pythonExecutable: join(userData, 'runtime', 'Scripts', 'python.exe'),
        runnerScript: join(resources, 'runtime', 'atomic_runtime.py'),
      })
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('runs a validated payload through the project-local Python runtime', async () => {
    const trace = await runAtomicRuntime({ kind: 'model', graph: gqaPreset })

    expect(trace.engine).toBe('pytorch')
    expect(trace.status).toBe('completed')
  })

  it('runs the complete GPT-like starter through PyTorch', async () => {
    const trace = await runAtomicRuntime({ kind: 'model', graph: gptLikeStarterPreset })

    expect(trace.status).toBe('completed')
    expect(trace.results).toHaveLength(gptLikeStarterPreset.nodes.length)
    expect(trace.results.at(-1)).toMatchObject({ atomId: 'head', status: 'passed' })
    expect(trace.modelOutput).toMatchObject({
      kind: 'logits',
      tensorShape: [2, 8, 32000],
      logitsShape: [2, 8, 32000],
      predictedTokenId: expect.any(Number),
      topTokenIds: expect.arrayContaining([expect.any(Number)]),
    })
  })

  it('runs agent-added dependencies in graph order rather than creation order', async () => {
    const lateIds = new Set(['head-layout', 'sdpa'])
    const graph = {
      ...gptLikeStarterPreset,
      nodes: [
        ...gptLikeStarterPreset.nodes.filter((node) => !lateIds.has(node.id)),
        ...gptLikeStarterPreset.nodes.filter((node) => lateIds.has(node.id)),
      ],
    }
    const trace = await runAtomicRuntime({ kind: 'model', graph, tokenIds: [1, 2, 3] })

    expect(trace.status).toBe('completed')
    expect(trace.results.find((result) => result.atomId === 'head-layout')).toMatchObject({ status: 'passed' })
    expect(trace.results.find((result) => result.atomId === 'sdpa')).toMatchObject({ status: 'passed' })
    expect(trace.results.find((result) => result.atomId === 'merge-heads')).toMatchObject({ status: 'passed' })
  })

  it.each([
    ['vision', visionTransformerPreset, [2, 8, 512]],
    ['multimodal image editing', multimodalImageEditorPreset, [2, 8, 512]],
    ['video', videoTransformerPreset, [2, 8, 512]],
    ['audio', audioEncoderPreset, [2, 18, 1024]],
  ] as const)('runs the %s starter through PyTorch', async (_name, graph, expectedShape) => {
    const trace = await runAtomicRuntime({ kind: 'model', graph })

    expect(trace.status).toBe('completed')
    expect(trace.results).toHaveLength(graph.nodes.length)
    expect(trace.results.at(-1)).toMatchObject({ status: 'passed' })
    expect(trace.modelOutput).toMatchObject({ kind: 'tensor', tensorShape: expectedShape })
  })

  it('automatically matches grouped-query KV heads at SDPA', async () => {
    const graph = {
      ...gptLikeStarterPreset,
      id: 'gpt-gqa-auto-match',
      config: { ...gptLikeStarterPreset.config, queryHeads: 8, keyValueHeads: 2, headDim: 96 },
    }
    const trace = await runAtomicRuntime({ kind: 'model', graph, tokenIds: [1, 2, 3] })

    expect(trace.status).toBe('completed')
    expect(trace.results.find((result) => result.atomId === 'sdpa')).toMatchObject({ status: 'passed' })
  })

  it('continues a healthy parallel architecture after another one fails', async () => {
    const graph: ArchitectureGraph = {
      id: 'parallel-runtime-isolation', name: 'Parallel runtime isolation', architecture: 'custom',
      config: { hiddenSize: 32, queryHeads: 2, keyValueHeads: 2, headDim: 16 },
      contracts: { causal: false, preservesGqaAtZeroGate: false, sdpaCompatible: false, contextualValue: false },
      nodes: [
        { id: 'broken-relu', kind: 'semantic', atomId: 'relu', label: 'Broken ReLU', role: 'hidden', position: { x: 0, y: 100 } },
        { id: 'healthy-input', kind: 'input', label: 'Healthy input', role: 'hidden', position: { x: 300, y: 0 } },
        { id: 'healthy-relu', kind: 'semantic', atomId: 'relu', label: 'Healthy ReLU', role: 'hidden', position: { x: 300, y: 100 } },
      ],
      edges: [{ id: 'healthy-edge', source: 'healthy-input', sourcePort: 'hidden', target: 'healthy-relu', targetPort: 'hidden' }],
    }
    const trace = await runAtomicRuntime({ kind: 'model', graph })

    expect(trace.status).toBe('failed')
    expect(trace.currentAtomId).toBe('broken-relu')
    expect(trace.results.find((result) => result.atomId === 'healthy-relu')).toMatchObject({ status: 'passed' })
  })

  it('returns a generic tensor output for a model without an LM head', async () => {
    const trace = await runAtomicRuntime({ kind: 'model', graph: trBasicPreset, tokenIds: [1, 2, 3] })

    expect(trace.status).toBe('completed')
    expect(trace.modelOutput).toEqual({ kind: 'tensor', tensorShape: [1, 3, 384] })
  })

  it('executes the complete learned MoE routing chain', async () => {
    const trace = await runAtomicRuntime({ kind: 'model', graph: tokenMoePreset, tokenIds: [1, 2, 3, 4] })

    expect(trace.status).toBe('completed')
    expect(trace.results).toHaveLength(tokenMoePreset.nodes.length)
    expect(trace.results.find((result) => result.atomId === 'router')).toMatchObject({ status: 'passed' })
    expect(trace.results.find((result) => result.atomId === 'topk')).toMatchObject({ status: 'passed' })
    expect(trace.results.find((result) => result.atomId === 'merge')).toMatchObject({ status: 'passed' })
    expect(trace.results.at(-1)).toMatchObject({ atomId: 'head', status: 'passed' })
    expect(trace.modelOutput).toMatchObject({ kind: 'logits', tensorShape: [1, 4, 32768] })
  })

  it('executes logits plus Training Labels through cross-entropy', async () => {
    const graph: ArchitectureGraph = {
      id: 'cross-entropy-runtime', name: 'Cross entropy', architecture: 'custom',
      config: { hiddenSize: 128, queryHeads: 2, keyValueHeads: 2, headDim: 64 },
      contracts: { causal: false, preservesGqaAtZeroGate: false, sdpaCompatible: false, contextualValue: false },
      nodes: [
        { id: 'logits', kind: 'input', label: 'Logits', role: 'logits', position: { x: 0, y: 0 } },
        { id: 'labels', kind: 'input', label: 'Training Labels', role: 'labels', position: { x: 100, y: 0 } },
        { id: 'loss', kind: 'semantic', atomId: 'cross-entropy-loss', label: 'Cross-entropy loss', role: 'scalar', position: { x: 50, y: 100 }, attributes: { ignoreIndex: -100 } },
      ],
      edges: [
        { id: 'logits-loss', source: 'logits', sourcePort: 'logits', target: 'loss', targetPort: 'logits' },
        { id: 'labels-loss', source: 'labels', sourcePort: 'labels', target: 'loss', targetPort: 'labels' },
      ],
    }
    const trace = await runAtomicRuntime({ kind: 'model', graph })

    expect(trace.status).toBe('completed')
    expect(trace.results.at(-1)).toMatchObject({ atomId: 'loss', status: 'passed', summary: 'shape=[] dtype=float32 finite=True' })
    expect(trace.modelOutput).toEqual({ kind: 'tensor', tensorShape: [] })
  })

  it('executes every activation card exposed by the block library', async () => {
    const activations = Object.values(activationAtomRegistry)
    const graph: ArchitectureGraph = {
      ...gqaPreset,
      id: 'activation-runtime-smoke',
      nodes: [
        { id: 'hidden-input', kind: 'input', label: 'Hidden input', role: 'hidden', position: { x: 0, y: 0 } },
        ...activations.map((definition, index) => ({
          id: `${definition.id}-test`,
          kind: 'semantic' as const,
          atomId: definition.id,
          label: definition.label,
          role: 'hidden' as const,
          position: { x: 0, y: (index + 1) * 100 },
          attributes: Object.fromEntries(definition.settings.map((setting) => [setting.id, setting.default])),
        })),
      ],
      edges: activations.map((definition, index) => ({
        id: `activation-${index}`,
        source: index === 0 ? 'hidden-input' : `${activations[index - 1].id}-test`,
        target: `${definition.id}-test`,
        sourcePort: 'output',
        targetPort: 'hidden',
      })),
      groups: [],
    }

    const trace = await runAtomicRuntime({ kind: 'model', graph })

    expect(trace.status).toBe('completed')
    expect(trace.results).toHaveLength(activations.length + 1)
  })

  it('executes a chain of newly catalogued normalization, MLP, and composition cards', async () => {
    const graph: ArchitectureGraph = {
      ...gqaPreset,
      id: 'extended-catalog-runtime',
      config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 2, headDim: 4 },
      groups: [],
      nodes: [
        { id: 'hidden', kind: 'input', label: 'Hidden', role: 'hidden', position: { x: 0, y: 0 } },
        { id: 'scale-norm', kind: 'semantic', atomId: 'scale-norm', label: 'ScaleNorm', role: 'hidden', position: { x: 0, y: 100 } },
        { id: 'tanh-mlp', kind: 'semantic', atomId: 'tanh-mlp', label: 'Tanh MLP', role: 'hidden', position: { x: 0, y: 200 }, attributes: { intermediateSize: 32, bias: true } },
        { id: 'clamp', kind: 'semantic', atomId: 'clamp', label: 'Clamp', role: 'hidden', position: { x: 0, y: 300 }, attributes: { minimum: -1, maximum: 1 } },
      ],
      edges: [
        { id: 'hidden-norm', source: 'hidden', target: 'scale-norm', sourcePort: 'output', targetPort: 'hidden' },
        { id: 'norm-mlp', source: 'scale-norm', target: 'tanh-mlp', sourcePort: 'output', targetPort: 'hidden' },
        { id: 'mlp-clamp', source: 'tanh-mlp', target: 'clamp', sourcePort: 'output', targetPort: 'hidden' },
      ],
    }

    const trace = await runAtomicRuntime({ kind: 'model', graph })
    expect(trace.status).toBe('completed')
    expect(trace.results.at(-1)).toMatchObject({ atomId: 'clamp', status: 'passed' })
    expect(trace.modelOutput).toEqual({ kind: 'tensor', tensorShape: [2, 8, 16] })
  })

  it('executes a safe user-created PyTorch card in the desktop runtime', async () => {
    const graph: ArchitectureGraph = {
      ...gqaPreset,
      id: 'custom-card-runtime',
      config: { hiddenSize: 16, queryHeads: 2, keyValueHeads: 2, headDim: 4 },
      groups: [],
      nodes: [
        { id: 'hidden', kind: 'input', label: 'Hidden', role: 'hidden', position: { x: 0, y: 0 } },
        { id: 'custom-norm', kind: 'custom-pytorch', label: 'My LayerNorm', role: 'hidden', position: { x: 0, y: 100 }, code: 'nn.LayerNorm(16)' },
      ],
      edges: [{ id: 'hidden-custom', source: 'hidden', target: 'custom-norm', sourcePort: 'output', targetPort: 'hidden' }],
    }

    const trace = await runAtomicRuntime({ kind: 'model', graph })
    expect(trace.status).toBe('completed')
    expect(trace.results.at(-1)).toMatchObject({ atomId: 'custom-norm', status: 'passed' })
  })

  it('rejects unknown runtime requests before spawning Python', async () => {
    await expect(runAtomicRuntime({ kind: 'shell' } as never)).rejects.toThrow('Unsupported atomic runtime kind')
  })
})
