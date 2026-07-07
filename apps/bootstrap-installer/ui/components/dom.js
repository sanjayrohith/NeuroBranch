export function createElement(tagName, options = {}, children = []) {
  const node = document.createElement(tagName)
  if (options.className) node.className = options.className
  if (options.text !== undefined) node.textContent = options.text
  for (const [name, value] of Object.entries(options.attributes || {})) {
    node.setAttribute(name, value)
  }
  node.append(...children.filter(Boolean))
  return node
}
