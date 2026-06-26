import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { delimiter, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export type AtomicRuntimePayload =
  | { kind: 'model'; graph: Record<string, unknown>; tokenIds?: number[] }
  | { kind: 'tokenizer'; pipeline: Record<string, unknown>; sample?: string }

export interface RuntimeAtomResult {
  atomId: string
  status: 'passed' | 'failed'
  summary?: string
  error?: string
}

export interface AtomicRuntimeTrace {
  engine: 'pytorch' | 'tokenizers'
  status: 'completed' | 'failed'
  currentAtomId?: string
  error?: string
  tokenIds?: number[]
  modelOutput?: {
    kind: 'logits' | 'tensor'
    tensorShape: number[]
    logitsShape?: number[]
    predictedTokenId?: number
    topTokenIds?: number[]
    topProbabilities?: number[]
  }
  results: RuntimeAtomResult[]
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const maximumPayloadBytes = 2 * 1024 * 1024
const maximumOutputBytes = 2 * 1024 * 1024

export interface AtomicRuntimePaths {
  pythonExecutable: string
  runnerScript: string
}

export interface AtomicRuntimePathOptions {
  projectRoot?: string
  resourcesPath?: string
  homeDirectory?: string
  userDataDirectory?: string
  configuredPython?: string
  environmentPath?: string
  platform?: NodeJS.Platform
}

export function resolveAtomicRuntimePaths(options: AtomicRuntimePathOptions = {}): AtomicRuntimePaths {
  const root = options.projectRoot ?? projectRoot
  const resources = options.resourcesPath ?? (typeof process.resourcesPath === 'string' ? process.resourcesPath : root)
  const home = options.homeDirectory ?? homedir()
  const platform = options.platform ?? process.platform
  const environmentPath = options.environmentPath ?? process.env.PATH ?? process.env.Path ?? ''
  const pathDelimiter = platform === 'win32' ? ';' : delimiter
  const executableNames = platform === 'win32' ? ['python.exe', 'python3.exe', 'py.exe'] : ['python3', 'python']
  const pathCandidates = environmentPath.split(pathDelimiter).filter(Boolean).flatMap((directory) => executableNames.map((name) => resolve(directory, name)))
  const pythonCandidates = [
    options.configuredPython,
    process.env.LABO_AI_PYTHON,
    options.userDataDirectory && resolve(options.userDataDirectory, 'runtime', 'Scripts', 'python.exe'),
    options.userDataDirectory && resolve(options.userDataDirectory, 'runtime', 'bin', 'python'),
    resolve(root, '.venv', 'Scripts', 'python.exe'),
    resolve(root, '.venv', 'bin', 'python'),
    resolve(home, 'Dev', 'labo-ai', '.venv', 'Scripts', 'python.exe'),
    resolve(home, 'Dev', 'labo-ai', '.venv', 'bin', 'python'),
    resolve(home, 'Development', 'labo-ai', '.venv', 'Scripts', 'python.exe'),
    resolve(home, 'Development', 'labo-ai', '.venv', 'bin', 'python'),
    resolve(home, 'Projects', 'labo-ai', '.venv', 'Scripts', 'python.exe'),
    resolve(home, 'Projects', 'labo-ai', '.venv', 'bin', 'python'),
    ...pathCandidates,
    '/opt/homebrew/bin/python3',
    '/usr/local/bin/python3',
    '/usr/bin/python3',
  ].filter((candidate): candidate is string => Boolean(candidate))
  const pythonExecutable = pythonCandidates.find(existsSync)
  if (!pythonExecutable) {
    throw new Error('Atomic runtime needs Python 3 with PyTorch. Set LABO_AI_PYTHON or keep a project .venv available (.venv/Scripts/python.exe on Windows).')
  }

  const runnerCandidates = [
    resolve(root, 'scripts', 'atomic_runtime.py'),
    resolve(resources, 'runtime', 'atomic_runtime.py'),
  ]
  const runnerScript = runnerCandidates.find(existsSync)
  if (!runnerScript) throw new Error('Atomic runtime script is missing from this LABO AI build.')
  return { pythonExecutable, runnerScript }
}

function validatePayload(payload: AtomicRuntimePayload): void {
  if (!payload || (payload.kind !== 'model' && payload.kind !== 'tokenizer')) {
    throw new Error('Unsupported atomic runtime kind')
  }
  if (payload.kind === 'model' && (!payload.graph || typeof payload.graph !== 'object')) {
    throw new Error('Model runtime requires a graph object')
  }
  if (payload.kind === 'tokenizer' && (!payload.pipeline || typeof payload.pipeline !== 'object')) {
    throw new Error('Tokenizer runtime requires a pipeline object')
  }
}

export async function runAtomicRuntime(payload: AtomicRuntimePayload, pathOptions: AtomicRuntimePathOptions = {}): Promise<AtomicRuntimeTrace> {
  validatePayload(payload)
  const input = JSON.stringify(payload)
  if (Buffer.byteLength(input) > maximumPayloadBytes) throw new Error('Atomic runtime payload is too large')
  const { pythonExecutable, runnerScript } = resolveAtomicRuntimePaths(pathOptions)

  return new Promise((resolvePromise, reject) => {
    const child = spawn(pythonExecutable, [runnerScript], {
      cwd: dirname(runnerScript),
      env: { ...process.env, PYTHONNOUSERSITE: '1' },
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error('Atomic runtime timed out'))
    }, 30_000)

    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      stdout += chunk
      if (Buffer.byteLength(stdout) > maximumOutputBytes) child.kill('SIGKILL')
    })
    child.stderr.on('data', (chunk: string) => { stderr += chunk })
    child.on('error', (error) => {
      clearTimeout(timer)
      reject(new Error(`Atomic runtime could not start: ${error.message}`))
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      if (code !== 0) {
        reject(new Error(stderr.trim() || `Atomic runtime exited with code ${code}`))
        return
      }
      try {
        resolvePromise(JSON.parse(stdout) as AtomicRuntimeTrace)
      } catch {
        reject(new Error(`Atomic runtime returned invalid JSON: ${stdout.slice(0, 200)}`))
      }
    })
    child.stdin.end(input)
  })
}
