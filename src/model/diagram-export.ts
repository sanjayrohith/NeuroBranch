import type { ArchitectureGraph, ArchitectureNode, TensorRole } from '../core/ir'

const cardWidth = 210
const cardHeight = 88

function xml(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;')
}

function roleColor(role: TensorRole): string {
  return ({ query: '#5797ff', key: '#d99a3d', value: '#8b73ff', logits: '#c678dd', attention: '#7c75ff', 'token-ids': '#39c887' } as Partial<Record<TensorRole, string>>)[role] ?? '#69717e'
}

function nodeColor(node: ArchitectureNode): string {
  if (node.kind === 'input') return '#39c887'
  if (node.kind === 'custom-pytorch') return '#62a9ff'
  if (node.role === 'logits' || node.role === 'output') return '#9b91ff'
  return roleColor(node.role)
}

export function architectureDiagramSvg(graph: ArchitectureGraph): string {
  if (graph.nodes.length === 0) return `<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420"><rect width="720" height="420" fill="#090a0c"/><text x="360" y="210" fill="#7d8490" text-anchor="middle" font-family="monospace">${xml(graph.name)} · blank graph</text></svg>`
  // Export the graph exactly as arranged in the editor. Re-layout here made
  // downloaded diagrams diverge from the composition the user had reviewed.
  const publicationGraph = graph
  const paddingX = 70
  const headerHeight = 66
  const minX = Math.min(...publicationGraph.nodes.map((node) => node.position.x))
  const maxX = Math.max(...publicationGraph.nodes.map((node) => node.position.x + cardWidth))
  const minY = Math.min(...publicationGraph.nodes.map((node) => node.position.y))
  const maxY = Math.max(...publicationGraph.nodes.map((node) => node.position.y + cardHeight))
  const width = Math.max(960, maxX - minX + paddingX * 2)
  const height = Math.max(540, maxY - minY + headerHeight + 55)
  const offsetX = (width - (maxX - minX)) / 2
  const position = (node: ArchitectureNode) => ({ x: node.position.x - minX + offsetX, y: node.position.y - minY + headerHeight })
  const byId = new Map(publicationGraph.nodes.map((node) => [node.id, node]))
  const portIds = (nodeId: string, direction: 'input' | 'output') => [...new Set(publicationGraph.edges
    .filter((edge) => direction === 'output' ? edge.source === nodeId : edge.target === nodeId)
    .map((edge) => direction === 'output' ? edge.sourcePort : edge.targetPort))]
  const portX = (node: ArchitectureNode, portId: string | undefined, direction: 'input' | 'output') => {
    const ports = portIds(node.id, direction)
    const index = Math.max(0, ports.indexOf(portId))
    return position(node).x + cardWidth * (index + 1) / (ports.length + 1)
  }
  const edges = publicationGraph.edges.map((edge) => {
    const source = byId.get(edge.source)
    const target = byId.get(edge.target)
    if (!source || !target) return ''
    const from = position(source)
    const to = position(target)
    const sx = portX(source, edge.sourcePort, 'output')
    const sy = from.y + cardHeight
    const tx = portX(target, edge.targetPort, 'input')
    const ty = to.y
    const middle = sy + (ty - sy) / 2
    const color = roleColor(source.role)
    return `<path d="M ${sx} ${sy} C ${sx} ${middle}, ${tx} ${middle}, ${tx} ${ty}" fill="none" stroke="${color}" stroke-width="2" opacity="0.82" marker-end="url(#arrow)"/>`
  }).join('')
  const nodes = publicationGraph.nodes.map((node) => {
    const point = position(node)
    const inputs = portIds(node.id, 'input').map((portId, index, ports) => `<circle cx="${cardWidth * (index + 1) / (ports.length + 1)}" cy="0" r="4" fill="#090a0c" stroke="${nodeColor(node)}"><title>${xml(portId ?? 'input')}</title></circle>`).join('')
    const outputs = portIds(node.id, 'output').map((portId, index, ports) => `<circle cx="${cardWidth * (index + 1) / (ports.length + 1)}" cy="${cardHeight}" r="4" fill="#090a0c" stroke="${nodeColor(node)}"><title>${xml(portId ?? 'output')}</title></circle>`).join('')
    return `<g transform="translate(${point.x} ${point.y})"><rect width="${cardWidth}" height="${cardHeight}" rx="10" fill="#15171c" stroke="${nodeColor(node)}" stroke-width="1.5"/>${inputs}${outputs}<text x="14" y="20" fill="#747c88" font-size="9" font-family="monospace" letter-spacing="0.6">${xml((node.atomId ?? node.kind).toUpperCase().slice(0, 30))}</text><text x="14" y="47" fill="#edf0f5" font-size="13" font-family="sans-serif" font-weight="600">${xml(node.label.slice(0, 30))}</text><text x="14" y="70" fill="#8b93a0" font-size="9" font-family="monospace">${xml(node.id)} · ${xml(node.role)}</text></g>`
  }).join('')
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><title>${xml(graph.name)}</title><defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 L 0 0 0 24" fill="none" stroke="#252932" stroke-width="0.6"/></pattern><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#77808d"/></marker></defs><rect width="100%" height="100%" fill="#090a0c"/><rect width="100%" height="100%" fill="url(#grid)"/><rect x="0" y="0" width="100%" height="48" fill="#111319" opacity=".96"/><text x="24" y="29" fill="#d4d8df" font-family="monospace" font-size="13" font-weight="600">${xml(graph.name)} · ${graph.nodes.length} cards · ${graph.edges.length} links</text>${edges}${nodes}</svg>`
}

export function exportFileName(graph: ArchitectureGraph, extension: 'svg' | 'py'): string {
  const base = graph.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'labo-architecture'
  return `${base}.${extension}`
}
