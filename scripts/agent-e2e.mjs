import { app, BrowserWindow, ipcMain } from 'electron'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { askLabo } from '../dist-electron/ask-labo.js'
import { getOpenAISettingsStatus } from '../dist-electron/openai-credentials.js'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const prompt = process.argv.slice(2).join(' ').trim() || 'From the Blank Starter, build a small executable DeepSeek-like language model: Token IDs, token embedding, normalization, learned top-k router, routed and shared experts in parallel, merge, final normalization, and tied LM head. Wire every compatible port. If a required unary hidden-state capability is unavailable, create a safe PyTorch card for it; otherwise identify the exact missing card.'

app.setName('LABO AI')

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function waitFor(window, expression, timeout = 60_000) {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    const value = await window.webContents.executeJavaScript(expression, true)
    if (value) return value
    await wait(250)
  }
  throw new Error(`Timed out waiting for: ${expression}`)
}

await app.whenReady()
ipcMain.handle('labo:ask', (_event, payload) => askLabo(payload))
ipcMain.handle('labo:openai-settings', () => getOpenAISettingsStatus())
const settings = await getOpenAISettingsStatus()
process.stderr.write(`[agent-e2e] credentials=${settings.configured ? settings.source : 'missing'}\n`)
if (!settings.configured) throw new Error('LABO AI has no configured OpenAI API key')

const window = new BrowserWindow({
  show: false,
  width: 1440,
  height: 900,
  webPreferences: {
    preload: join(projectRoot, 'dist-electron', 'preload.cjs'),
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
    partition: 'agent-e2e',
  },
})

try {
  process.stderr.write('[agent-e2e] loading renderer\n')
  await window.loadFile(join(projectRoot, 'dist', 'index.html'))
  await waitFor(window, `document.querySelector('button[aria-label="Play model atoms"]') !== null`)
  await window.webContents.executeJavaScript(`(() => {
    const button = [...document.querySelectorAll('button')].find((candidate) => candidate.textContent.trim() === 'Blank starter')
    if (!button) throw new Error('Blank Starter button not found')
    button.click()
    const ask = [...document.querySelectorAll('button')].find((candidate) => candidate.textContent.includes('Ask LABO'))
    if (!ask) throw new Error('Ask LABO button not found')
    ask.click()
  })()`, true)
  await waitFor(window, `document.querySelector('.ask-labo-key-heading')?.textContent.includes('Connected') === true`)
  process.stderr.write('[agent-e2e] Ask LABO connected\n')
  await window.webContents.executeJavaScript(`(() => {
    const textarea = document.querySelector('#ask-labo-request')
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, ${JSON.stringify(prompt)})
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  })()`, true)
  await waitFor(window, `[...document.querySelectorAll('button')].some((button) => button.textContent.includes('Propose graph changes') && !button.disabled)`)
  process.stderr.write('[agent-e2e] submitting DeepSeek-like request\n')
  await window.webContents.executeJavaScript(`[...document.querySelectorAll('button')].find((button) => button.textContent.includes('Propose graph changes')).click()`, true)
  await waitFor(window, `document.querySelector('.ask-labo-result, .ask-labo-error') !== null`, 90_000)
  process.stderr.write('[agent-e2e] plan received\n')

  const preview = await window.webContents.executeJavaScript(`(() => ({
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
  }))()`, true)

  if (preview.canApply) {
    await window.webContents.executeJavaScript(`document.querySelector('.ask-labo-apply').click()`, true)
    await waitFor(window, `document.querySelector('.ask-labo-panel') === null`)
  }

  const graph = await window.webContents.executeJavaScript(`(() => ({
    status: document.querySelector('.statusbar')?.textContent.trim(),
    cards: [...document.querySelectorAll('.architecture-node')].map((element) => ({
      id: element.dataset.nodeId,
      atomId: element.dataset.atomId,
      label: element.querySelector('strong')?.textContent.trim(),
      x: Number.parseFloat(element.style.left),
      y: Number.parseFloat(element.style.top),
    })),
    edges: document.querySelectorAll('[data-edge-id]').length,
  }))()`, true)

  process.stdout.write(`${JSON.stringify({ prompt, preview, graph }, null, 2)}\n`)
  if (preview.error) process.exitCode = 1
} finally {
  window.destroy()
  app.quit()
}
