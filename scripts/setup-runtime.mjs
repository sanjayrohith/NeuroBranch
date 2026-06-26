import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const requirementsPath = join(projectRoot, 'requirements-runtime.txt')
const targetFlagIndex = process.argv.indexOf('--venv')

if (targetFlagIndex !== -1 && !process.argv[targetFlagIndex + 1]) {
  throw new Error('--venv requires a target directory')
}

const virtualEnvironment = resolve(targetFlagIndex === -1 ? join(projectRoot, '.venv') : process.argv[targetFlagIndex + 1])
const virtualPython = process.platform === 'win32'
  ? join(virtualEnvironment, 'Scripts', 'python.exe')
  : join(virtualEnvironment, 'bin', 'python')
const stampPath = join(virtualEnvironment, '.labo-runtime-requirements.sha256')
const requiredImports = 'import torch, tokenizers, typing_extensions, jinja2, setuptools'

function run(command, args, options = {}) {
  return spawnSync(command, args, {
    cwd: projectRoot,
    encoding: 'utf8',
    windowsHide: true,
    ...options,
  })
}

function usablePython(command, prefixArgs = []) {
  const probe = run(command, [...prefixArgs, '-c', 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'])
  return probe.status === 0 ? { command, prefixArgs } : undefined
}

function findPython() {
  const configured = process.env.LABO_AI_PYTHON
  const candidates = configured
    ? [[configured, []]]
    : process.platform === 'win32'
      ? [['py', ['-3']], ['python', []], ['python3', []]]
      : [['python3', []], ['python', []]]

  for (const [command, prefixArgs] of candidates) {
    const candidate = usablePython(command, prefixArgs)
    if (candidate) return candidate
  }
  throw new Error('Python 3.10 or newer is required to prepare the LABO AI runtime. Set LABO_AI_PYTHON to a compatible interpreter.')
}

function runOrThrow(command, args, description) {
  const result = run(command, args, { stdio: 'inherit', encoding: undefined })
  if (result.error) throw new Error(`${description}: ${result.error.message}`)
  if (result.status !== 0) throw new Error(`${description} exited with code ${result.status ?? 'unknown'}`)
}

function importsAreAvailable() {
  if (!existsSync(virtualPython)) return false
  return run(virtualPython, ['-c', requiredImports]).status === 0
}

const requirements = await readFile(requirementsPath)
const fingerprint = createHash('sha256').update(requirements).digest('hex')
const previousFingerprint = existsSync(stampPath) ? (await readFile(stampPath, 'utf8')).trim() : ''

if (previousFingerprint === fingerprint && importsAreAvailable()) {
  console.log(`LABO AI Python runtime is ready at ${virtualEnvironment}`)
  process.exit(0)
}

if (!existsSync(virtualPython)) {
  const basePython = findPython()
  console.log(`Creating the LABO AI Python runtime at ${virtualEnvironment}`)
  await mkdir(dirname(virtualEnvironment), { recursive: true })
  runOrThrow(basePython.command, [...basePython.prefixArgs, '-m', 'venv', virtualEnvironment], 'Python virtual-environment creation')
}

console.log('Installing the locked LABO AI Python runtime dependencies...')
runOrThrow(virtualPython, ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', requirementsPath], 'Python dependency installation')

if (!importsAreAvailable()) {
  throw new Error('The LABO AI Python runtime was installed, but torch or tokenizers could not be imported.')
}

await writeFile(stampPath, `${fingerprint}\n`, 'utf8')
console.log(`LABO AI Python runtime is ready at ${virtualEnvironment}`)
