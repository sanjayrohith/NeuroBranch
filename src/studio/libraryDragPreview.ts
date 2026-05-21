export function setLibraryDragPreview(dataTransfer: DataTransfer, label: string, glyphClass: string) {
  if (typeof dataTransfer.setDragImage !== 'function') return

  const preview = document.createElement('div')
  preview.className = 'library-drag-preview'
  const glyph = document.createElement('span')
  const text = document.createElement('span')
  glyph.className = `block-glyph ${glyphClass}`
  text.textContent = label
  preview.append(glyph, text)
  document.body.append(preview)
  dataTransfer.setDragImage(preview, 18, 18)
  window.setTimeout(() => preview.remove(), 0)
}
