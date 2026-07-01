const endpoint = process.argv[2] || 'http://127.0.0.1:9334/json/list'
const parallel = process.argv.slice(3).includes('--parallel')
const demo = process.argv.slice(3).includes('--demo')
const prepare = process.argv.slice(3).includes('--prepare')
const playPrompt = process.argv.slice(3).find((argument) => argument.startsWith('--play-prompt='))?.slice('--play-prompt='.length) || 'What is a neural network?'
const prompt = process.argv.slice(3).filter((argument) => !['--parallel', '--demo', '--prepare'].includes(argument) && !argument.startsWith('--play-prompt=')).join(' ').trim()
  || (parallel ? 'Add a small executable GPT-like language model as a separate parallel architecture. Keep the existing graph-wide dimensions.' : 'Build a small executable DeepSeek-like language model from the Blank Starter. Keep the existing graph-wide dimensions and do not edit card settings.')

const pages = await fetch(endpoint).then((response) => response.json())
const page = pages.find((candidate) => candidate.type === 'page' && candidate.title === 'LABO AI')
if (!page?.webSocketDebuggerUrl) throw new Error('No LABO AI renderer was found on the debugging endpoint')

const socket = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true })
  socket.addEventListener('error', reject, { once: true })
})

let sequence = 0
const pending = new Map()
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data)
  if (!message.id || !pending.has(message.id)) return
  const { resolve, reject } = pending.get(message.id)
  pending.delete(message.id)
  if (message.error) reject(new Error(message.error.message))
  else resolve(message.result)
})

function send(method, params = {}) {
  const id = ++sequence
  socket.send(JSON.stringify({ id, method, params }))
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
}

async function evaluate(expression) {
  const response = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description || response.exceptionDetails.text)
  return response.result?.value
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function waitFor(expression, timeout = 60_000) {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    const value = await evaluate(expression)
    if (value) return value
    await wait(250)
  }
  throw new Error(`Timed out waiting for: ${expression}`)
}

await send('Runtime.enable')
await waitFor(`document.querySelector('button[aria-label="Play model atoms"]') !== null`)
if (prepare) {
  await evaluate(`(() => {
    const exactButton = (text) => [...document.querySelectorAll('button')].find((button) => button.textContent.trim() === text)
    exactButton('Blocks')?.click()
    exactButton('Blank starter')?.click()
  })()`)
  await wait(500)
  await evaluate(`document.querySelector('.reset-model-preset-button')?.click()`)
  await waitFor(`document.querySelector('.reset-model-preset-button')?.textContent.includes('Confirm restore') === true`)
  await evaluate(`document.querySelector('.reset-model-preset-button')?.click()`)
  await waitFor(`document.querySelectorAll('.architecture-node').length === 0`)
  await evaluate(`[...document.querySelectorAll('button')].find((button) => button.textContent.includes('Ask LABO'))?.click()`)
  await waitFor(`document.querySelector('#ask-labo-request') !== null`)
  await evaluate(`(() => {
    const textarea = document.querySelector('#ask-labo-request')
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, '')
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    document.querySelector('button[aria-label="Close Ask LABO"]')?.click()
  })()`)
  await waitFor(`document.querySelector('.ask-labo-panel') === null`)
  await evaluate(`(() => {
    const panel = (label) => [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes(label))
    if (panel('Library')?.getAttribute('aria-pressed') !== 'true') panel('Library')?.click()
    if (panel('Inspector')?.getAttribute('aria-pressed') === 'true') panel('Inspector')?.click()
  })()`)
  const state = await evaluate(`({
    cards: document.querySelectorAll('.architecture-node').length,
    status: document.querySelector('.statusbar')?.textContent.trim(),
    blocksView: [...document.querySelectorAll('.view-switcher button')].find((button) => button.textContent.trim() === 'Blocks')?.getAttribute('aria-pressed'),
    libraryOpen: [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes('Library'))?.getAttribute('aria-pressed'),
    inspectorOpen: [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes('Inspector'))?.getAttribute('aria-pressed'),
  })`)
  process.stdout.write(`${JSON.stringify({ prepared: true, state }, null, 2)}\n`)
  socket.close()
  process.exit(0)
}
await evaluate(`(() => {
  const input = document.querySelector('input[aria-label="Model generation prompt"]')
  if (!input) return
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
  setter.call(input, ${JSON.stringify(playPrompt)})
  input.dispatchEvent(new Event('input', { bubbles: true }))
})()`)
await evaluate(`(() => {
  const exactButton = (text) => [...document.querySelectorAll('button')].find((button) => button.textContent.trim() === text)
  exactButton(${JSON.stringify(parallel ? 'TR 300M' : 'Blank starter')})?.click()
  const askButton = [...document.querySelectorAll('button')].find((button) => button.textContent.includes('Ask LABO'))
  askButton?.click()
})()`)
await waitFor(`document.querySelector('.ask-labo-key-heading')?.textContent.includes('Connected') === true`)
if (demo) await wait(2500)
const beforeCards = await evaluate(`[...document.querySelectorAll('.architecture-node')].map((element) => ({ atomId: element.dataset.atomId, label: element.querySelector('strong')?.textContent.trim(), x: Number.parseFloat(element.style.left), y: Number.parseFloat(element.style.top) }))`)
await evaluate(`(() => {
  const review = [...document.querySelectorAll('button')].find((button) => button.textContent.trim() === 'Review')
  review?.click()
  if (${parallel}) [...document.querySelectorAll('button')].find((button) => button.textContent.trim() === 'New parallel')?.click()
  const textarea = document.querySelector('#ask-labo-request')
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
  setter.call(textarea, ${JSON.stringify(prompt)})
  textarea.dispatchEvent(new Event('input', { bubbles: true }))
})()`)
if (demo) await wait(3500)
await waitFor(`[...document.querySelectorAll('button')].some((button) => button.textContent.includes('Propose graph changes') && !button.disabled)`)
await evaluate(`[...document.querySelectorAll('button')].find((button) => button.textContent.includes('Propose graph changes')).click()`)
await waitFor(`document.querySelector('.ask-labo-result, .ask-labo-error') !== null`, 210_000)
if (demo) await wait(5000)

const preview = await evaluate(`(() => ({
  error: document.querySelector('.ask-labo-error')?.textContent.trim() ?? null,
  summary: document.querySelector('.ask-labo-result section p')?.textContent.trim() ?? null,
  addedBlocks: [...document.querySelectorAll('.ask-labo-added-blocks > div')].map((element) => ({
    nodeId: element.querySelector('strong')?.textContent.trim(),
    atomId: element.querySelector('code')?.textContent.trim(),
    reason: element.querySelector('small')?.textContent.trim(),
  })),
  createdBlocks: [...document.querySelectorAll('.ask-labo-created-blocks > div')].map((element) => ({
    label: element.querySelector('strong')?.textContent.trim(),
    pytorchModule: element.querySelector('code')?.textContent.trim(),
    reason: element.querySelector('small')?.textContent.trim(),
  })),
  missingBlocks: [...document.querySelectorAll('.ask-labo-missing > div')].map((element) => ({
    label: element.querySelector('strong')?.textContent.trim(),
    reason: element.querySelector('small')?.textContent.trim(),
  })),
  warnings: [...document.querySelectorAll('.ask-labo-warnings p')].map((element) => element.textContent.trim()),
  connectionCount: document.querySelectorAll('.ask-labo-connections li').length,
  canApply: !document.querySelector('.ask-labo-apply')?.disabled,
}))()`)

let graph = null
let player = null
  if (preview.canApply) {
  await evaluate(`document.querySelector('.ask-labo-apply').click()`)
  await waitFor(`document.querySelector('.ask-labo-panel') === null`)
  if (demo) await wait(5000)
  if (demo) {
    await evaluate(`(() => {
      const panel = (label) => [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes(label))
      if (panel('Library')?.getAttribute('aria-pressed') === 'true') panel('Library')?.click()
      if (panel('Inspector')?.getAttribute('aria-pressed') === 'true') panel('Inspector')?.click()
      document.querySelector('button[aria-label="Fit graph"]')?.click()
    })()`)
    await wait(4000)
  }
  graph = await evaluate(`(() => ({
    status: document.querySelector('.statusbar')?.textContent.trim(),
    compileStatus: document.querySelector('.toolbar-meta > span')?.textContent.trim(),
    cards: [...document.querySelectorAll('.architecture-node')].map((element) => ({
      id: element.dataset.nodeId,
      atomId: element.dataset.atomId,
      label: element.querySelector('strong')?.textContent.trim(),
      x: Number.parseFloat(element.style.left),
      y: Number.parseFloat(element.style.top),
    })),
    edges: document.querySelectorAll('[data-edge-id]').length,
  }))()`)

  if (graph.status?.includes('Neural IR valid')) {
    if (demo) {
      await evaluate(`document.querySelector('.model-prompt-menu')?.setAttribute('open', '')`)
      await wait(3000)
      await evaluate(`document.querySelector('.model-prompt-menu')?.removeAttribute('open')`)
      await wait(1000)
    }
    await evaluate(`document.querySelector('button[aria-label="Play model atoms"]').click()`)
    await waitFor(`['completed', 'failed'].includes(document.querySelector('.player-status')?.textContent.trim())`, 90_000)
    player = await evaluate(`(() => ({
      status: document.querySelector('.player-status')?.textContent.trim(),
      error: document.querySelector('.execution-error')?.textContent.trim() ?? null,
      output: document.querySelector('[aria-label="Model generation output"]')?.textContent.trim(),
    }))()`)
  }

  if (demo) await wait(6000)

  if (demo && player?.status === 'completed') {
    await evaluate(`[...document.querySelectorAll('.view-switcher button')].find((button) => button.textContent.trim() === 'PyTorch')?.click()`)
    await wait(5000)
    await evaluate(`[...document.querySelectorAll('.view-switcher button')].find((button) => button.textContent.trim() === 'Split')?.click()`)
    await wait(5000)
    await evaluate(`[...document.querySelectorAll('.interaction-switcher button')].find((button) => button.textContent.trim() === 'Edit cards')?.click()`)
    await wait(2500)
    await evaluate(`document.querySelector('.architecture-node[data-atom-id="qkv-projection"] .node-select')?.click()`)
    await waitFor(`document.querySelector('[aria-label="Edit model card"]') !== null`)
    await wait(4500)
    await evaluate(`document.querySelector('button[aria-label="Close model card editor"]')?.click()`)
    await wait(2000)
    await evaluate(`[...document.querySelectorAll('.view-switcher button')].find((button) => button.textContent.trim() === 'Blocks')?.click()`)
    await wait(3500)
  }

  if (!demo) {
    await evaluate(`(() => {
      const reset = [...document.querySelectorAll('button')].find((button) => button.textContent.trim().startsWith('Restore '))
      reset?.click()
    })()`)
  }
}

const existingPreserved = graph ? beforeCards.every((card, index) => JSON.stringify(card) === JSON.stringify(graph.cards[index])) : null
process.stdout.write(`${JSON.stringify({ prompt, playPrompt, parallel, existingPreserved, preview, graph, player }, null, 2)}\n`)
socket.close()
