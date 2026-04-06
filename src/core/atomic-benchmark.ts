export interface AtomicBenchmarkResult {
  contractId: string
  implementationId: string
  correctness: {
    forwardMaxAbsError: number
    backwardMaxAbsError: number
    finite: boolean
    deterministic: boolean
  }
  performance: {
    latencyUs: number
    memoryBytes: number
    allocations: number
  }
}

export interface AtomicTolerances {
  forwardTolerance: number
  backwardTolerance: number
}

export function createAtomicBenchmarkResult(result: AtomicBenchmarkResult): AtomicBenchmarkResult {
  return structuredClone(result)
}

function percentDelta(baseline: number, candidate: number): number {
  if (baseline === 0) return candidate === 0 ? 0 : Number.POSITIVE_INFINITY
  return ((candidate - baseline) / baseline) * 100
}

export function compareAtomicImplementations(
  baseline: AtomicBenchmarkResult,
  candidate: AtomicBenchmarkResult,
  tolerances: AtomicTolerances,
) {
  if (baseline.contractId !== candidate.contractId) {
    throw new Error('Cannot compare different atomic contracts')
  }

  const candidatePassesContract =
    candidate.correctness.finite
    && candidate.correctness.deterministic
    && candidate.correctness.forwardMaxAbsError <= tolerances.forwardTolerance
    && candidate.correctness.backwardMaxAbsError <= tolerances.backwardTolerance

  const latencyPercent = percentDelta(
    baseline.performance.latencyUs,
    candidate.performance.latencyUs,
  )
  const memoryPercent = percentDelta(
    baseline.performance.memoryBytes,
    candidate.performance.memoryBytes,
  )
  const allocationsPercent = percentDelta(
    baseline.performance.allocations,
    candidate.performance.allocations,
  )

  const noPerformanceRegression = latencyPercent <= 0 && memoryPercent <= 0 && allocationsPercent <= 0
  const strictPerformanceGain = latencyPercent < 0 || memoryPercent < 0 || allocationsPercent < 0
  const verdict = !candidatePassesContract
    ? 'baseline-better'
    : noPerformanceRegression && strictPerformanceGain
      ? 'candidate-better'
      : 'inconclusive'

  return {
    contractId: baseline.contractId,
    baselineImplementationId: baseline.implementationId,
    candidateImplementationId: candidate.implementationId,
    candidatePassesContract,
    delta: {
      latencyPercent,
      memoryPercent,
      allocationsPercent,
      forwardMaxAbsError: candidate.correctness.forwardMaxAbsError - baseline.correctness.forwardMaxAbsError,
      backwardMaxAbsError: candidate.correctness.backwardMaxAbsError - baseline.correctness.backwardMaxAbsError,
    },
    verdict,
  }
}
