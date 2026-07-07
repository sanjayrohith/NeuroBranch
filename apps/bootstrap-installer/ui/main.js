import { createBrandHeader } from './components/brand-header.js'
import { createHeroSection } from './components/hero-section.js'
import { createInstallFooter } from './components/install-footer.js'
import { createProgressCard } from './components/progress-card.js'

const root = document.querySelector('#setup-app')
const status = createProgressCard()
const footer = createInstallFooter()

root.append(createBrandHeader(), createHeroSection(), status.element, footer.element)

function setInstalling(installing) {
  footer.button.disabled = installing
  footer.button.textContent = installing ? 'Installing...' : 'Install latest'
  root.classList.toggle('is-installing', installing)
}

function showPreview() {
  status.setVersion('Latest release')
  status.update({ stage: 'Runtime', message: 'Preparing the isolated Python runtime...', percent: 45 })
  status.appendLog('5% Release', 'Checking the latest LABO AI release...')
  status.appendLog('12% Source', 'Downloading the latest release source...')
  status.appendLog('36% Dependencies', 'Installing locked JavaScript dependencies...')
  status.appendLog('45% Runtime', 'Preparing the isolated Python runtime...')
  setInstalling(true)
}

async function startTauriIntegration() {
  const tauri = window.__TAURI__
  if (!tauri?.core?.invoke || !tauri?.event?.listen) {
    showPreview()
    return
  }

  const invoke = tauri.core.invoke
  const listen = tauri.event.listen

  async function refresh() {
    try {
      const state = await invoke('setup_status')
      status.setVersion(state.latestTag ? `Latest ${state.latestTag}` : `Installed ${state.installedTag || 'none'}`)
      footer.button.disabled = state.installing
      footer.button.textContent = state.installing
        ? 'Installing...'
        : state.installedTag === state.latestTag
          ? 'Reinstall latest'
          : state.installedTag
            ? 'Update LABO AI'
            : 'Install latest'
    } catch (error) {
      status.setVersion('GitHub check unavailable')
      status.setMessage(String(error))
    }
  }

  await listen('setup-progress', ({ payload }) => {
    const failed = payload.stage === 'Failed'
    setInstalling(!failed)
    footer.button.textContent = failed ? 'Retry installation' : 'Installing...'
    status.update(payload)
    status.appendLog(`${payload.percent}% ${payload.stage}`, payload.message)
  })

  footer.button.addEventListener('click', async () => {
    setInstalling(true)
    try {
      const result = await invoke('install_latest')
      if (result.setupRelaunched) {
        status.setVersion(result.tag)
        status.update({
          stage: 'Updating Setup',
          message: 'The verified latest Setup is taking over this installation.',
        })
        status.appendLog('Relaunch', 'Closing this Setup and continuing in the newly verified helper.')
        return
      }
      status.setVersion(result.tag)
      status.update({
        stage: 'Installed',
        message: 'LABO AI is installed and launching. Future updates are available from Settings.',
        percent: 100,
      })
      footer.button.textContent = 'Installed'
    } catch (error) {
      const detail = String(error)
      if (detail.includes('already installing')) {
        status.update({
          stage: 'Installing',
          message: 'The active installation is still running in this Setup window.',
        })
      } else {
        status.update({ stage: 'Failed', message: detail })
        setInstalling(false)
        footer.button.textContent = 'Retry installation'
      }
    }
  })

  await refresh()
}

void startTauriIntegration()
