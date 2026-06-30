import { spawn } from 'node:child_process'
import { access, readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const require = createRequire(import.meta.url)
const { extractFile } = require('@electron/asar')

function check(condition, message) {
  if (!condition) throw new Error(message)
  process.stdout.write(`✓ ${message}\n`)
}

async function exists(path) {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

async function runPythonSmoke() {
  const python = join(root, '.venv', 'bin', 'python')
  const runner = join(root, 'scripts', 'atomic_runtime.py')
  const payload = JSON.stringify({
    kind: 'model',
    graph: {
      config: { hiddenSize: 32, queryHeads: 2, keyValueHeads: 1, headDim: 16 },
      nodes: [{ id: 'hidden', kind: 'input', label: 'Hidden state', role: 'hidden', position: { x: 0, y: 0 } }],
      edges: [],
    },
  })
  const output = await new Promise((resolve, reject) => {
    const child = spawn(python, [runner], { cwd: root, stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.on('error', reject)
    child.on('close', (code) => code === 0 ? resolve(stdout) : reject(new Error(stderr || `Python exited with ${code}`)))
    child.stdin.end(payload)
  })
  const trace = JSON.parse(output)
  check(trace.status === 'completed' && trace.results?.[0]?.status === 'passed', 'PyTorch atomic runtime executes a desktop payload')
}

const html = await readFile(join(root, 'dist', 'index.html'), 'utf8')
check(html.includes('src="./assets/') && !html.includes('src="/assets/'), 'Vite assets use file://-safe relative paths')
check(await exists(join(root, 'dist-electron', 'main.js')), 'Electron main process is compiled')
check(await exists(join(root, 'dist-electron', 'preload.cjs')), 'Electron preload bridge is compiled')
check(await exists(join(root, 'build', 'icon.icns')), 'macOS application icon is present')
await runPythonSmoke()

for (const app of [
  join(root, 'release', 'mac-arm64', 'LABO AI.app'),
  '/Applications/LABO AI.app',
]) {
  const archive = join(app, 'Contents', 'Resources', 'app.asar')
  if (!await exists(archive)) continue
  const packagedHtml = extractFile(archive, 'dist/index.html').toString('utf8')
  check(packagedHtml.includes('src="./assets/'), `${app} contains the non-black-screen renderer build`)
  check(await exists(join(app, 'Contents', 'Resources', 'runtime', 'atomic_runtime.py')), `${app} contains the Python runner resource`)
}

for (const app of [
  join(root, 'release', 'win-unpacked'),
  join(root, 'release', 'win-arm64-unpacked'),
]) {
  const archive = join(app, 'resources', 'app.asar')
  if (!await exists(archive)) continue
  const packagedHtml = extractFile(archive, 'dist/index.html').toString('utf8')
  check(packagedHtml.includes('src="./assets/'), `${app} contains the non-black-screen renderer build`)
  check(await exists(join(app, 'resources', 'runtime', 'atomic_runtime.py')), `${app} contains the Python runner resource`)
  check(await exists(join(app, 'LABO AI.exe')), `${app} contains the Windows executable`)
}
