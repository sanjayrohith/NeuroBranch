import { describe, expect, it } from 'vitest'
import { compileToPyTorch } from './ir'
import { parsePyTorchDialect } from './pytorch-dialect'
import { gqaPreset, tokenMoePreset } from './presets'

describe('LABO AI PyTorch dialect', () => {
  it('updates the same IR atom when supported PyTorch settings change', () => {
    const source = compileToPyTorch(gqaPreset).replace(
      'self.q_proj = nn.Linear(384, 384, bias=False)',
      'self.q_proj = nn.Linear(384, 384, bias=True)',
    )
    const result = parsePyTorchDialect(source, gqaPreset)

    expect(result.diagnostics).toEqual([])
    expect(result.graph.nodes.find((node) => node.id === 'q-proj')?.attributes).toMatchObject({
      inFeatures: 384,
      outFeatures: 384,
      bias: true,
    })
  })

  it('removes a managed block when its canonical declaration is deleted', () => {
    const source = compileToPyTorch(gqaPreset).replace(
      /\s*# labo:node=q-proj kind=linear\n\s*self\.q_proj = nn\.Linear\([^\n]+\)\n/,
      '\n',
    )
    const result = parsePyTorchDialect(source, gqaPreset)

    expect(result.graph.nodes.some((node) => node.id === 'q-proj')).toBe(false)
    expect(result.validation.valid).toBe(false)
    expect(result.validation.errors).toContain('SDPA requires a connected query input')
  })

  it('removes an elastic from the IR when its managed PyTorch edge is deleted', () => {
    const source = compileToPyTorch(gqaPreset).replace(
      /^\s*# labo:edge=k-sdpa source=k-proj target=sdpa[^\n]*\n/m,
      '',
    )
    const result = parsePyTorchDialect(source, gqaPreset)

    expect(result.graph.edges.some((edge) => edge.id === 'k-sdpa')).toBe(false)
    expect(result.graph.edges).toHaveLength(gqaPreset.edges.length - 1)
  })

  it('rewires an elastic when its managed PyTorch edge endpoints change', () => {
    const source = compileToPyTorch(gqaPreset).replace(
      '# labo:edge=hidden-q source=hidden target=q-proj',
      '# labo:edge=hidden-q source=output target=q-proj',
    )
    const result = parsePyTorchDialect(source, gqaPreset)

    expect(result.graph.edges.find((edge) => edge.id === 'hidden-q')).toMatchObject({ source: 'output', target: 'q-proj' })
  })

  it('updates semantic atom settings from its registry-managed declaration', () => {
    const source = compileToPyTorch(tokenMoePreset).replace(
      'self.norm = nn.RMSNorm(384, eps=1e-06)',
      'self.norm = nn.RMSNorm(384, eps=1e-05)',
    )
    const result = parsePyTorchDialect(source, tokenMoePreset)
    expect(result.diagnostics).toEqual([])
    expect(result.graph.nodes.find((node) => node.id === 'norm')?.attributes?.epsilon).toBe(1e-5)
  })

  it('adds a new semantic atom from a recognized LABO marker and declaration', () => {
    const source = compileToPyTorch(tokenMoePreset).replace(
      '    def forward(',
      '        # labo:node=extra_relu atom=relu\n        self.extra_relu = nn.ReLU(inplace=False)\n\n    def forward(',
    )
    const result = parsePyTorchDialect(source, tokenMoePreset)
    expect(result.diagnostics).toEqual([])
    expect(result.graph.nodes.find((node) => node.id === 'extra_relu')).toMatchObject({ kind: 'semantic', atomId: 'relu' })
  })

  it('removes a semantic atom and its incident elastics when its marker is deleted', () => {
    const source = compileToPyTorch(tokenMoePreset).replace(
      /\s*# labo:node=router atom=moe-router\n\s*self\.router = nn\.Linear\([^\n]+\)\n/,
      '\n',
    )
    const result = parsePyTorchDialect(source, tokenMoePreset)
    expect(result.graph.nodes.some((node) => node.id === 'router')).toBe(false)
    expect(result.graph.edges.some((edge) => edge.source === 'router' || edge.target === 'router')).toBe(false)
  })
})
