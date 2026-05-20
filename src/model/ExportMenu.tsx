import { Download } from 'lucide-react'
import type { ArchitectureGraph } from '../core/ir'
import { exportArchitectureDiagram, exportPyTorchCode } from './export-actions'

export function ExportMenu({ graph, codeGraph = graph, code, onDiagram, onPyTorch }: { graph: ArchitectureGraph; codeGraph?: ArchitectureGraph; code: string; onDiagram?: () => unknown; onPyTorch?: () => unknown }) {
  const close = (target: HTMLElement) => target.closest('details')?.removeAttribute('open')
  const diagram = onDiagram ?? (() => exportArchitectureDiagram(graph))
  const pytorch = onPyTorch ?? (() => exportPyTorchCode(codeGraph, code))
  return <details className="export-menu">
    <summary aria-label="Export architecture"><Download size={13} /><span>Export</span></summary>
    <div>
      <button onClick={(event) => { close(event.currentTarget); void diagram() }}><strong>Diagram SVG</strong><small>Vector Blockly graph</small></button>
      <button onClick={(event) => { close(event.currentTarget); void pytorch() }}><strong>PyTorch code</strong><small>Generated .py module</small></button>
      <button onClick={(event) => { close(event.currentTarget); void (async () => { await diagram(); await pytorch() })() }}><strong>Diagram + code</strong><small>Export both files</small></button>
    </div>
  </details>
}
