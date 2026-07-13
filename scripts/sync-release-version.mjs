import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const rootPackage = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'))
const version = rootPackage.version

if (!/^\d+\.\d+\.\d+$/.test(version)) {
  throw new Error(`Unsupported release version: ${version}`)
}

function updateJson(relativePath) {
  const path = resolve(root, relativePath)
  const value = JSON.parse(readFileSync(path, 'utf8'))
  value.version = version

  if (value.packages?.['']) {
    value.packages[''].version = version
  }

  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`)
}

function replacePackageVersion(relativePath, packageName) {
  const path = resolve(root, relativePath)
  const source = readFileSync(path, 'utf8')
  const packagePattern = new RegExp(`(name = "${packageName}"\\r?\\nversion = ")[^"]+("\\r?\\n)`)
  const next = source.replace(packagePattern, `$1${version}$2`)

  if (next === source) {
    throw new Error(`Could not update ${packageName} in ${relativePath}`)
  }

  writeFileSync(path, next)
}

updateJson('apps/bootstrap-installer/package.json')
updateJson('apps/bootstrap-installer/package-lock.json')
updateJson('apps/bootstrap-installer/src-tauri/tauri.conf.json')
replacePackageVersion('apps/bootstrap-installer/src-tauri/Cargo.toml', 'labo-ai-setup')
replacePackageVersion('apps/bootstrap-installer/src-tauri/Cargo.lock', 'labo-ai-setup')

console.log(`Synchronized LABO AI release version ${version}`)
