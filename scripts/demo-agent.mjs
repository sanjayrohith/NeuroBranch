import { app, BrowserWindow, ipcMain, shell } from 'electron'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { askLabo } from '../dist-electron/ask-labo.js'
import { runAtomicRuntime } from '../dist-electron/atomic-runtime.js'
import { CodexAppServer } from '../dist-electron/chatgpt-session.js'
import { getOpenAISettingsStatus } from '../dist-electron/openai-credentials.js'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const keepOpen = process.argv.includes('--keep-open')
const hideCues = process.argv.includes('--no-cues')
const buildPrompt = 'Build a compact executable GPT-like question-answering decoder from Token IDs to generated tokens. Use grouped-query causal attention, one residual SwiGLU MLP, final RMSNorm, a tied language-model head, and greedy token decoding. Wire every compatible typed port and arrange the graph clearly.'
const upgradePrompt = 'Upgrade the current architecture to a token-routed residual mixture of experts. Preserve the causal attention and output path. Replace the single residual MLP with one shared dense expert plus deterministic Token-ID routing over four routed residual experts, merge their outputs, reconnect every typed port, arrange the graph clearly, save it as Agent Demo, and validate it by running the atoms.'
const generationPrompt = 'Explain neural networks in one short sentence.'
const memoryState = new Map()

app.setName('LABO AI')
process.stderr.write('[demo] entry loaded\n')

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function waitFor(window, expression, timeout = 210_000) {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    const value = await window.webContents.executeJavaScript(expression, true)
    if (value) return value
    await wait(250)
  }
  throw new Error(`Demo timed out waiting for: ${expression}`)
}

async function evaluate(window, expression) {
  return window.webContents.executeJavaScript(expression, true)
}

async function clickButton(window, label, { exact = true } = {}) {
  const clicked = await evaluate(window, `(() => {
    const label = ${JSON.stringify(label)}
    const button = [...document.querySelectorAll('button')].find((candidate) => {
      const text = candidate.textContent.trim()
      const ariaLabel = candidate.getAttribute('aria-label') ?? ''
      return ${exact ? 'text === label || ariaLabel === label' : 'text.includes(label) || ariaLabel.includes(label)'}
    })
    if (!button || button.disabled) return false
    button.click()
    return true
  })()`)
  if (!clicked) throw new Error(`Demo button is unavailable: ${label}`)
}

async function setTextarea(window, value, typingDelay = 0) {
  await evaluate(window, `(async () => {
    const textarea = document.querySelector('textarea[aria-label="What should these blocks build?"]')
    if (!textarea) throw new Error('LABO agent prompt is unavailable')
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    const value = ${JSON.stringify(value)}
    if (${typingDelay} > 0) {
      for (let index = 1; index <= value.length; index += 3) {
        setter.call(textarea, value.slice(0, index))
        textarea.dispatchEvent(new Event('input', { bubbles: true }))
        await new Promise((resolve) => setTimeout(resolve, ${typingDelay}))
      }
    }
    setter.call(textarea, value)
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    textarea.focus()
  })()`)
}

async function cue(window, eyebrow, title, detail, duration = 3200) {
  if (hideCues) {
    await wait(duration)
    return
  }
  await evaluate(window, `(() => {
    let cue = document.querySelector('#labo-demo-cue')
    if (!cue) {
      cue = document.createElement('div')
      cue.id = 'labo-demo-cue'
      cue.style.cssText = 'position:fixed;z-index:9999;right:24px;top:96px;width:390px;padding:14px 16px;border:1px solid rgba(111,211,176,.42);border-radius:12px;background:linear-gradient(135deg,rgba(11,22,20,.96),rgba(20,18,32,.96));box-shadow:0 18px 60px rgba(0,0,0,.55);pointer-events:none;font-family:Inter,sans-serif'
      document.body.append(cue)
    }
    cue.innerHTML = '<small style="display:block;margin-bottom:5px;color:#7fdcb9;font:9px JetBrains Mono;letter-spacing:.16em;text-transform:uppercase">' + ${JSON.stringify(eyebrow)} + '</small><strong style="display:block;color:#eff7f4;font-size:15px">' + ${JSON.stringify(title)} + '</strong><span style="display:block;margin-top:6px;color:#97a6a1;font:10px/1.45 JetBrains Mono">' + ${JSON.stringify(detail)} + '</span>'
  })()`)
  await wait(duration)
}

async function prepareBlankWorkspace(window) {
  await clickButton(window, 'Blocks')
  await evaluate(window, `document.querySelector('.preset-menu > summary')?.click()`)
  await clickButton(window, 'Blank starter')
  await wait(600)

  if (await evaluate(window, `document.querySelectorAll('.architecture-node').length > 0`)) {
    await clickButton(window, 'Settings')
    await clickButton(window, 'Workspaces')
    await clickButton(window, 'Restore')
    await clickButton(window, 'Confirm restore')
    await clickButton(window, 'Close LABO AI settings')
  }

  await waitFor(window, `document.querySelectorAll('.architecture-node').length === 0`)
  await evaluate(window, `(() => {
    const panel = (label) => [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes(label))
    if (panel('Library')?.getAttribute('aria-pressed') !== 'true') panel('Library')?.click()
    if (panel('Inspector')?.getAttribute('aria-pressed') === 'true') panel('Inspector')?.click()
  })()`)
}

async function selectReviewMode(window) {
  await clickButton(window, 'Settings')
  await clickButton(window, 'Agent')
  await clickButton(window, 'Review')
  await clickButton(window, 'Extend current')
  await clickButton(window, 'Close LABO AI settings')
}

async function clearAgentActivity(window) {
  const activityOpen = await evaluate(window, `document.querySelector('[aria-label="Agent activity"]') !== null`)
  if (!activityOpen) {
    await clickButton(window, 'Open agent activity')
    await waitFor(window, `document.querySelector('[aria-label="Agent activity"]') !== null`)
  }
  await clickButton(window, 'Clear')
  if (await evaluate(window, `document.querySelector('[aria-label="Agent activity"]') !== null`)) {
    await clickButton(window, 'Close agent activity')
  }
  await waitFor(window, `document.querySelector('[aria-label="Agent activity"]') === null`)
}

async function runDemo() {
process.stderr.write('[demo] electron ready\n')
const chatGPT = new CodexAppServer(
  (url) => shell.openExternal(url),
  app.getVersion(),
  process.env.LABO_CODEX_HOME?.trim() || join(app.getPath('userData'), 'codex'),
)
const providerStatus = Promise.all([
  getOpenAISettingsStatus(),
  chatGPT.status().catch(() => ({ available: false, connected: false })),
]).then(([openAISettings, chatGPTStatus]) => ({ openAISettings, chatGPTStatus }))

ipcMain.handle('labo:ask', async (_event, payload) => {
  const { chatGPTStatus } = await providerStatus
  return chatGPTStatus.connected ? chatGPT.ask(payload, {}) : askLabo(payload)
})
ipcMain.handle('labo:atomic-runtime', (_event, payload) => runAtomicRuntime(payload))
ipcMain.handle('labo:openai-settings', async () => (await providerStatus).openAISettings)
ipcMain.handle('labo:chatgpt-session', async () => (await providerStatus).chatGPTStatus)
ipcMain.handle('labo:desktop-state-load', (_event, payload) => memoryState.get(payload?.scope))
ipcMain.handle('labo:desktop-state-save', (_event, payload) => { memoryState.set(payload?.scope, payload?.data); return { saved: true } })
ipcMain.handle('labo:window-state', (event) => ({
  fullScreen: BrowserWindow.fromWebContents(event.sender)?.isFullScreen() ?? false,
}))

const window = new BrowserWindow({
  show: false,
  width: 1440,
  height: 900,
  backgroundColor: '#08090b',
  title: 'LABO AI · Agent demo',
  autoHideMenuBar: process.platform !== 'darwin',
  ...(process.platform === 'darwin' ? { titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 15, y: 17 } } : {}),
  webPreferences: {
    preload: join(projectRoot, 'dist-electron', 'preload.cjs'),
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
    partition: 'labo-agent-demo',
  },
})
process.stderr.write('[demo] browser window created\n')
const publishWindowState = () => window.webContents.send('labo:window-state', { fullScreen: window.isFullScreen() })
window.on('enter-full-screen', publishWindowState)
window.on('leave-full-screen', publishWindowState)

try {
  process.stderr.write('[demo] loading renderer\n')
  await window.loadFile(join(projectRoot, 'dist', 'index.html'))
  process.stderr.write('[demo] renderer loaded\n')
  window.show()
  if (process.platform === 'darwin') {
    const enteredFullScreen = new Promise((resolve) => window.once('enter-full-screen', resolve))
    window.setFullScreen(true)
    await enteredFullScreen
  } else {
    window.maximize()
  }
  window.focus()
  process.stderr.write('[demo] window visible; waiting 5 seconds before agent actions\n')
  await wait(5_000)
  const { openAISettings, chatGPTStatus } = await providerStatus
  if (!openAISettings.configured && !chatGPTStatus.connected) throw new Error('Connect ChatGPT or add and verify an OpenAI API key in LABO AI before running the agent demo.')
  process.stderr.write(`[demo] provider=${chatGPTStatus.connected ? 'chatgpt' : openAISettings.source}\n`)
  await waitFor(window, `document.querySelector('button[aria-label="Play model atoms"]') !== null`)
  await prepareBlankWorkspace(window)
  await selectReviewMode(window)
  await cue(window, 'LABO AI', 'From conversation to executable graph', 'One agent, typed tools, inspectable PyTorch.', 3800)

  process.stderr.write('[demo] sending conversational greeting\n')
  await setTextarea(window, 'Hello', 85)
  await clickButton(window, 'Propose graph changes')
  await waitFor(window, `document.querySelector('[aria-label="Agent activity"] li[data-status="answered"]') !== null`)
  await cue(window, '1 · Conversation', 'Hello.', 'The same prompt understands ordinary conversation and graph-building requests.', 1400)
  await clearAgentActivity(window)

  process.stderr.write('[demo] submitting graph build request\n')
  await setTextarea(window, buildPrompt, 7)
  await cue(window, '2 · Natural-language brief', 'Build a token-routed GPT-like QA decoder', 'LABO receives the live graph, typed ports and the complete card catalog.', 3600)
  await clickButton(window, 'Propose graph changes')
  await waitFor(window, `document.querySelector('.agent-plan-review, .ask-labo-error') !== null`, 210_000)
  const agentError = await evaluate(window, `document.querySelector('.ask-labo-error')?.textContent.trim() ?? ''`)
  if (agentError) throw new Error(agentError)
  await evaluate(window, `document.querySelector('.agent-plan-review-content')?.scrollTo({ top: 0 })`)
  await cue(window, '3 · Auditable plan', 'Review cards, elastics and tool calls', 'Nothing mutates until the complete locally validated plan is approved.', 7200)

  await clickButton(window, 'Apply full plan')
  await waitFor(window, `document.querySelector('.agent-plan-review') === null && document.querySelectorAll('.architecture-node').length >= 12`)
  await clearAgentActivity(window)
  await evaluate(window, `(() => {
    const panel = (label) => [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes(label))
    if (panel('Library')?.getAttribute('aria-pressed') === 'true') panel('Library')?.click()
    if (panel('Inspector')?.getAttribute('aria-pressed') === 'true') panel('Inspector')?.click()
    document.querySelector('button[aria-label="Fit graph"]')?.click()
  })()`)
  await cue(window, '4 · First architecture', 'A compact executable GPT-like decoder', 'The approved baseline is already a valid typed graph.', 4800)

  process.stderr.write('[demo] submitting architecture upgrade request\n')
  await setTextarea(window, upgradePrompt, 7)
  await cue(window, '5 · Iterative architecture design', 'Upgrade the current graph to token-routed MoE', 'The agent must preserve valid paths while replacing and reconnecting the residual MLP.', 4200)
  await clickButton(window, 'Propose graph changes')
  await waitFor(window, `document.querySelector('.agent-plan-review, .ask-labo-error') !== null`, 210_000)
  const upgradeError = await evaluate(window, `document.querySelector('.ask-labo-error')?.textContent.trim() ?? ''`)
  if (upgradeError) throw new Error(upgradeError)
  await evaluate(window, `document.querySelector('.agent-plan-review-content')?.scrollTo({ top: 0 })`)
  await cue(window, '6 · Upgrade plan', 'Inspect replacements, deletions and new elastics', 'The second plan edits the existing architecture instead of rebuilding blindly.', 7200)

  await clickButton(window, 'Apply full plan')
  await waitFor(window, `document.querySelector('.agent-plan-review') === null && document.querySelectorAll('.architecture-node').length >= 16`)
  await clearAgentActivity(window)
  await evaluate(window, `(() => {
    const input = document.querySelector('input[aria-label="Model generation prompt"]')
    if (input) {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
      setter.call(input, ${JSON.stringify(generationPrompt)})
      input.dispatchEvent(new Event('input', { bubbles: true }))
    }
    const panel = (label) => [...document.querySelectorAll('.panel-visibility-button')].find((button) => button.textContent.includes(label))
    if (panel('Library')?.getAttribute('aria-pressed') === 'true') panel('Library')?.click()
    if (panel('Inspector')?.getAttribute('aria-pressed') === 'true') panel('Inspector')?.click()
    document.querySelector('button[aria-label="Fit graph"]')?.click()
  })()`)
  await cue(window, '7 · Upgraded architecture', 'Sequence, forks and expert joins remain visible', 'The XY engine arranges the upgraded graph while typed elastics preserve tensor contracts.', 6800)

  const status = await evaluate(window, `document.querySelector('.player-status')?.textContent.trim() ?? ''`)
  if (!['playing', 'completed'].includes(status)) await clickButton(window, 'Play model atoms')
  await cue(window, '8 · Local execution', 'Run the atoms, not a mock-up', 'The desktop player executes PyTorch and keeps every atomic result inspectable.', 4800)
  await waitFor(window, `['completed', 'failed'].includes(document.querySelector('.player-status')?.textContent.trim())`, 90_000)
  const finalStatus = await evaluate(window, `document.querySelector('.player-status')?.textContent.trim()`)
  if (finalStatus !== 'completed') {
    const executionError = await evaluate(window, `document.querySelector('.execution-error')?.textContent.trim() ?? 'Atomic execution failed'`)
    throw new Error(executionError)
  }
  await cue(window, 'Completed', 'Generated-token output is ready', 'The graph can now be rerun, reset or stepped atom by atom.', 5200)

  await clickButton(window, 'PyTorch')
  await cue(window, '9 · Synchronized PyTorch', 'The visual graph remains real code', 'Every supported card maps to inspectable generated PyTorch.', 5800)
  await clickButton(window, 'Split')
  await cue(window, '10 · One source of truth', 'Graph and code stay synchronized', 'Save the workspace, compare architectures, or export SVG and Python.', 5800)

  await clickButton(window, 'Open agent activity')
  await cue(window, 'Agent activity', 'Conversation and graph tools in one trace', 'The final history exposes accepted operations, validation and tool usage.', 6500)
  await clickButton(window, 'Close agent activity')
  await waitFor(window, `document.querySelector('[aria-label="Agent activity"]') === null`)
  await clickButton(window, 'Split')
  await cue(window, 'Final view', 'Typed graph and synchronized PyTorch', 'The demo ends on the complete architecture and its executable source.', 2800)
  await evaluate(window, `document.querySelector('#labo-demo-cue')?.remove()`)
  process.stderr.write('[demo] complete; edit API waiting time down to a short jump cut\n')
  if (keepOpen) await new Promise(() => undefined)
  await wait(2200)
} finally {
  if (!window.isDestroyed()) window.destroy()
  chatGPT.stop()
  app.quit()
}
}

app.whenReady().then(runDemo).catch((error) => {
  process.stderr.write(`[demo] failed: ${error?.stack ?? error}\n`)
  app.quit()
})
