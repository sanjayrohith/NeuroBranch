import type { AtomSetting, ModelAtomDefinition } from './model-atoms'

const hiddenInput = { id: 'hidden', tensor: 'hidden' as const, rank: 3 }
const hiddenOutput = { id: 'output', tensor: 'hidden' as const, rank: 3 }

function activation(id: string, label: string, expression: string, settings: AtomSetting[] = []): ModelAtomDefinition {
  const constructor = settings.reduce(
    (result, setting) => result.replaceAll(`=${setting.id}`, `={{${setting.id}}}`),
    expression,
  )
  return {
    id,
    label,
    category: 'activation',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings,
    lowerings: {
      pytorch: {
        executable: true,
        declarations: [`self.{{module}} = ${constructor}`],
        forward: ['{{out:output}} = self.{{module}}({{in:hidden}})'],
      },
    },
  }
}

export const activationAtomRegistry: Record<string, ModelAtomDefinition> = {
  relu: activation('relu', 'ReLU', 'nn.ReLU(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  relu6: activation('relu6', 'ReLU6', 'nn.ReLU6(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  gelu: activation('gelu', 'GELU', 'nn.GELU(approximate=approximate)', [{ id: 'approximate', type: 'select', default: 'none', options: ['none', 'tanh'] }]),
  silu: activation('silu', 'SiLU / Swish', 'nn.SiLU(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  sigmoid: activation('sigmoid', 'Sigmoid', 'nn.Sigmoid()'),
  tanh: activation('tanh', 'Tanh', 'nn.Tanh()'),
  softplus: activation('softplus', 'Softplus', 'nn.Softplus(beta=beta, threshold=threshold)', [{ id: 'beta', type: 'number', default: 1 }, { id: 'threshold', type: 'number', default: 20 }]),
  softsign: activation('softsign', 'Softsign', 'nn.Softsign()'),
  elu: activation('elu', 'ELU', 'nn.ELU(alpha=alpha, inplace=inplace)', [{ id: 'alpha', type: 'number', default: 1 }, { id: 'inplace', type: 'boolean', default: false }]),
  celu: activation('celu', 'CELU', 'nn.CELU(alpha=alpha, inplace=inplace)', [{ id: 'alpha', type: 'number', default: 1 }, { id: 'inplace', type: 'boolean', default: false }]),
  selu: activation('selu', 'SELU', 'nn.SELU(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  'leaky-relu': activation('leaky-relu', 'LeakyReLU', 'nn.LeakyReLU(negative_slope=negativeSlope, inplace=inplace)', [{ id: 'negativeSlope', type: 'number', default: 0.01 }, { id: 'inplace', type: 'boolean', default: false }]),
  prelu: activation('prelu', 'PReLU', 'nn.PReLU(num_parameters=numParameters, init=init)', [{ id: 'numParameters', type: 'number', default: 1 }, { id: 'init', type: 'number', default: 0.25 }]),
  rrelu: activation('rrelu', 'RReLU', 'nn.RReLU(lower=lower, upper=upper, inplace=inplace)', [{ id: 'lower', type: 'number', default: 0.125 }, { id: 'upper', type: 'number', default: 0.3333333333 }, { id: 'inplace', type: 'boolean', default: false }]),
  mish: activation('mish', 'Mish', 'nn.Mish(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  hardsigmoid: activation('hardsigmoid', 'Hardsigmoid', 'nn.Hardsigmoid(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  hardswish: activation('hardswish', 'Hardswish', 'nn.Hardswish(inplace=inplace)', [{ id: 'inplace', type: 'boolean', default: false }]),
  hardtanh: activation('hardtanh', 'Hardtanh', 'nn.Hardtanh(min_val=minValue, max_val=maxValue, inplace=inplace)', [{ id: 'minValue', type: 'number', default: -1 }, { id: 'maxValue', type: 'number', default: 1 }, { id: 'inplace', type: 'boolean', default: false }]),
  hardshrink: activation('hardshrink', 'Hardshrink', 'nn.Hardshrink(lambd=lambd)', [{ id: 'lambd', type: 'number', default: 0.5 }]),
  softshrink: activation('softshrink', 'Softshrink', 'nn.Softshrink(lambd=lambd)', [{ id: 'lambd', type: 'number', default: 0.5 }]),
  tanhshrink: activation('tanhshrink', 'Tanhshrink', 'nn.Tanhshrink()'),
  threshold: activation('threshold', 'Threshold', 'nn.Threshold(threshold=threshold, value=value, inplace=inplace)', [{ id: 'threshold', type: 'number', default: 0 }, { id: 'value', type: 'number', default: 0 }, { id: 'inplace', type: 'boolean', default: false }]),
  logsigmoid: activation('logsigmoid', 'LogSigmoid', 'nn.LogSigmoid()'),
  glu: activation('glu', 'GLU', 'nn.GLU(dim=dim)', [{ id: 'dim', type: 'number', default: -1 }]),
}

export interface GatedActivationDefinition {
  id: 'swiglu' | 'geglu' | 'reglu'
  label: string
  gateActivation: 'silu' | 'gelu' | 'relu'
  settings: AtomSetting[]
  primitives: ['gate-projection', 'value-projection', 'gate-activation', 'multiply', 'output-projection']
}

export const gatedActivationRegistry: Record<GatedActivationDefinition['id'], GatedActivationDefinition> = {
  swiglu: { id: 'swiglu', label: 'SwiGLU', gateActivation: 'silu', settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }], primitives: ['gate-projection', 'value-projection', 'gate-activation', 'multiply', 'output-projection'] },
  geglu: { id: 'geglu', label: 'GEGLU', gateActivation: 'gelu', settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }], primitives: ['gate-projection', 'value-projection', 'gate-activation', 'multiply', 'output-projection'] },
  reglu: { id: 'reglu', label: 'ReGLU', gateActivation: 'relu', settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }], primitives: ['gate-projection', 'value-projection', 'gate-activation', 'multiply', 'output-projection'] },
}
