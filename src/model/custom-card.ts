import type { ArchitectureGraph, TensorRole } from '../core/ir'

export interface CustomPyTorchCard {
  id: string
  label: string
  code: string
  inputRole?: TensorRole
  outputRole?: TensorRole
  /** A real reusable subgraph. `code` remains for legacy unary cards and previews. */
  graph?: ArchitectureGraph
}

export type CustomCardOperation = 'linear' | 'rmsnorm' | 'layernorm' | 'dropout' | 'gelu' | 'silu' | 'relu' | 'sigmoid' | 'tanh' | 'mish' | 'identity'
export type CustomCardCategory = 'projection' | 'normalization' | 'activation' | 'regularization' | 'utility'

export const customCardOperations: Array<{ id: CustomCardOperation; label: string }> = [
  { id: 'linear', label: 'Linear' },
  { id: 'rmsnorm', label: 'RMSNorm' },
  { id: 'layernorm', label: 'LayerNorm' },
  { id: 'dropout', label: 'Dropout' },
  { id: 'gelu', label: 'GELU' },
  { id: 'silu', label: 'SiLU' },
  { id: 'relu', label: 'ReLU' },
  { id: 'sigmoid', label: 'Sigmoid' },
  { id: 'tanh', label: 'Tanh' },
  { id: 'mish', label: 'Mish' },
  { id: 'identity', label: 'Identity' },
]

export const operationsByCategory: Record<CustomCardCategory, CustomCardOperation[]> = {
  projection: ['linear'],
  normalization: ['rmsnorm', 'layernorm'],
  activation: ['gelu', 'silu', 'relu', 'sigmoid', 'tanh', 'mish'],
  regularization: ['dropout'],
  utility: ['identity'],
}

export interface CustomCardComposition {
  operation: CustomCardOperation
  label: string
  code: string
  inputRole: TensorRole
  outputRole: TensorRole
}

export function customCardModule(operation: CustomCardOperation, inFeatures: number, outFeatures: number, probability: number): string {
  if (operation === 'linear') return `nn.Linear(${inFeatures}, ${outFeatures})`
  if (operation === 'rmsnorm') return `nn.RMSNorm(${outFeatures})`
  if (operation === 'layernorm') return `nn.LayerNorm(${outFeatures})`
  if (operation === 'dropout') return `nn.Dropout(${probability})`
  if (operation === 'gelu') return 'nn.GELU()'
  if (operation === 'silu') return 'nn.SiLU()'
  if (operation === 'relu') return 'nn.ReLU()'
  if (operation === 'sigmoid') return 'nn.Sigmoid()'
  if (operation === 'tanh') return 'nn.Tanh()'
  if (operation === 'mish') return 'nn.Mish()'
  return 'nn.Identity()'
}

export function suggestedCardOperation(category: CustomCardCategory, need: string): CustomCardOperation {
  const normalized = need.toLowerCase()
  if (category === 'normalization') return normalized.includes('layer') ? 'layernorm' : 'rmsnorm'
  if (category === 'activation') {
    if (normalized.includes('silu') || normalized.includes('swiglu')) return 'silu'
    if (normalized.includes('relu')) return 'relu'
    if (normalized.includes('sigmoid')) return 'sigmoid'
    if (normalized.includes('tanh')) return 'tanh'
    if (normalized.includes('mish')) return 'mish'
    return 'gelu'
  }
  if (category === 'regularization') return 'dropout'
  if (category === 'utility') return 'identity'
  return 'linear'
}

export function suggestedCardCategory(need: string): CustomCardCategory {
  const normalized = need.toLowerCase()
  if (/rms|layer.?norm|normaliz/.test(normalized)) return 'normalization'
  if (/gelu|silu|swiglu|relu|sigmoid|tanh|mish|activat/.test(normalized)) return 'activation'
  if (/dropout|regulari/.test(normalized)) return 'regularization'
  if (/identity|pass.?through|no.?op/.test(normalized)) return 'utility'
  return 'projection'
}

export function composeCustomCard({ category, need, inFeatures = 768, outFeatures = 768, probability = 0.1, inputRole, outputRole }: {
  category: CustomCardCategory
  need: string
  inFeatures?: number
  outFeatures?: number
  probability?: number
  inputRole?: TensorRole
  outputRole?: TensorRole
}): CustomCardComposition {
  const operation = suggestedCardOperation(category, need)
  const inferredRole: TensorRole = /logit|vocab|classifier|language head/i.test(need) ? 'logits' : 'hidden'
  const composedInputRole = inputRole ?? (category === 'projection' ? 'hidden' : inferredRole)
  const composedOutputRole = outputRole ?? inferredRole
  return {
    operation,
    label: need.trim().split(/[.!?]/)[0]?.slice(0, 42) || customCardOperations.find((candidate) => candidate.id === operation)?.label || 'Custom atom',
    code: customCardModule(operation, inFeatures, outFeatures, probability),
    inputRole: composedInputRole,
    outputRole: composedOutputRole,
  }
}
