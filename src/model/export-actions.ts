import type { ArchitectureGraph } from '../core/ir'
import { architectureDiagramSvg, exportFileName } from './diagram-export'

async function saveFile(filename: string, content: string, kind: 'svg' | 'python') {
  if (window.labo?.exportFile) return window.labo.exportFile({ filename, content, kind })
  const blob = new Blob([content], { type: kind === 'svg' ? 'image/svg+xml' : 'text/x-python' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
  return { saved: true }
}

export function exportArchitectureDiagram(graph: ArchitectureGraph) {
  return saveFile(exportFileName(graph, 'svg'), architectureDiagramSvg(graph), 'svg')
}

export function exportPyTorchCode(graph: ArchitectureGraph, code: string) {
  return saveFile(exportFileName(graph, 'py'), code, 'python')
}
