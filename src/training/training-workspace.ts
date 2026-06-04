import type { OptimizerConfig, OptimizerDefinition, OptimizerValue } from '../core/optimizer-ir'

export interface TrainingWorkspaceState {
  config: OptimizerConfig
  customOptimizers: OptimizerDefinition[]
  updatedAt: number
}

function validValue(value: unknown): value is OptimizerValue {
  if (value === null || ['number', 'boolean', 'string'].includes(typeof value)) return true
  return Array.isArray(value) && value.every((item) => item === null || typeof item === 'number')
}

function validSettings(value: unknown): value is Record<string, OptimizerValue> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) && Object.values(value).every(validValue)
}

function validDefinition(value: unknown): value is OptimizerDefinition {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<OptimizerDefinition>
  return typeof candidate.id === 'string'
    && typeof candidate.label === 'string'
    && typeof candidate.torchClass === 'string'
    && validSettings(candidate.defaults)
}

export function parseTrainingWorkspace(value: unknown): TrainingWorkspaceState | undefined {
  if (!value || typeof value !== 'object') return undefined
  const candidate = value as Partial<TrainingWorkspaceState>
  const customOptimizers = Array.isArray(candidate.customOptimizers) ? candidate.customOptimizers.filter(validDefinition) : []
  if (!candidate.config || typeof candidate.config.id !== 'string' || typeof candidate.config.kind !== 'string' || !validSettings(candidate.config.settings)) return undefined
  const available = new Set(['adamw', 'adam', 'muon', 'sgd', 'adafactor', 'rmsprop', 'nadam', 'radam', 'sparse-adam', 'adagrad', 'adadelta', 'asgd', 'lbfgs', ...customOptimizers.map((optimizer) => optimizer.id)])
  if (!available.has(candidate.config.kind)) return undefined
  return {
    config: candidate.config,
    customOptimizers,
    updatedAt: typeof candidate.updatedAt === 'number' && Number.isFinite(candidate.updatedAt) ? candidate.updatedAt : 0,
  }
}
