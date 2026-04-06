import { describe, expect, it } from 'vitest'
import { compareAtomicImplementations, createAtomicBenchmarkResult } from './atomic-benchmark'

const reference = createAtomicBenchmarkResult({
  contractId: 'rms-norm-v1',
  implementationId: 'torch-reference',
  correctness: {
    forwardMaxAbsError: 0,
    backwardMaxAbsError: 0,
    finite: true,
    deterministic: true,
  },
  performance: { latencyUs: 42, memoryBytes: 1_048_576, allocations: 4 },
})

const candidate = createAtomicBenchmarkResult({
  contractId: 'rms-norm-v1',
  implementationId: 'fused-candidate',
  correctness: {
    forwardMaxAbsError: 1.2e-6,
    backwardMaxAbsError: 2.8e-6,
    finite: true,
    deterministic: true,
  },
  performance: { latencyUs: 29, memoryBytes: 860_000, allocations: 2 },
})

describe('standalone atomic benchmark', () => {
  it('identifies a better implementation of the same atomic contract without training', () => {
    const comparison = compareAtomicImplementations(reference, candidate, {
      forwardTolerance: 1e-5,
      backwardTolerance: 1e-5,
    })

    expect(comparison.contractId).toBe('rms-norm-v1')
    expect(comparison.candidatePassesContract).toBe(true)
    expect(comparison.delta.latencyPercent).toBeCloseTo(-30.952, 2)
    expect(comparison.delta.memoryPercent).toBeLessThan(0)
    expect(comparison.verdict).toBe('candidate-better')
    expect('trainingMetrics' in comparison).toBe(false)
  })

  it('refuses to rank implementations of different semantic contracts', () => {
    const incompatible = { ...candidate, contractId: 'layer-norm-v1' }
    expect(() => compareAtomicImplementations(reference, incompatible, {
      forwardTolerance: 1e-5,
      backwardTolerance: 1e-5,
    })).toThrow('Cannot compare different atomic contracts')
  })
})
