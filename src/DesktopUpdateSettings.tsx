import { CheckCircle2, Download, FlaskConical, RefreshCw, ShieldCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

type Status = Awaited<ReturnType<NonNullable<NonNullable<Window['labo']>['getDesktopUpdateStatus']>>>

export function DesktopUpdateSettings() {
  const [statuses, setStatuses] = useState<Partial<Record<DesktopUpdateChannel, Status>>>({})
  const [channel, setChannel] = useState<DesktopUpdateChannel>('stable')
  const [busy, setBusy] = useState<'check' | 'launch' | ''>('')
  const [error, setError] = useState('')
  const [checked, setChecked] = useState(false)
  const statusRequest = useRef(0)
  const api = window.labo
  const status = statuses[channel]

  useEffect(() => {
    if (api?.runtime !== 'electron' || !api.getDesktopUpdateStatus) return
    const request = ++statusRequest.current
    api.getDesktopUpdateStatus().then((next) => {
      if (request !== statusRequest.current) return
      setStatuses((current) => ({ ...current, [next.channel]: next }))
      setChannel(next.channel)
    }).catch((cause) => {
      if (request === statusRequest.current) setError(cause instanceof Error ? cause.message : String(cause))
    })
  }, [api])

  if (api?.runtime !== 'electron') return null

  const check = async () => {
    if (!api.getDesktopUpdateStatus) {
      setError('This desktop build does not expose the update service.')
      return
    }
    setBusy('check')
    setError('')
    const request = ++statusRequest.current
    try {
      const next = await api.getDesktopUpdateStatus(channel)
      if (request !== statusRequest.current) return
      setStatuses((current) => ({ ...current, [channel]: next }))
      setChecked(true)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy('')
    }
  }

  const start = async () => {
    setBusy('launch')
    setError('')
    try {
      if (!status?.helperInstalled) {
        await api.openDesktopSetup?.()
        setBusy('')
        return
      }
      await api.launchDesktopUpdate?.(channel)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      setBusy('')
    }
  }

  const selectChannel = async (nextChannel: DesktopUpdateChannel) => {
    if (nextChannel === channel || !api.getDesktopUpdateStatus) return
    setChannel(nextChannel)
    setBusy('check')
    setChecked(false)
    setError('')
    const request = ++statusRequest.current
    try {
      const next = await api.getDesktopUpdateStatus(nextChannel)
      if (request !== statusRequest.current) return
      setStatuses((current) => ({ ...current, [nextChannel]: next }))
      setChecked(true)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy('')
    }
  }

  const installedRef = status?.installedTag ?? (status?.currentVersion ? `v${status.currentVersion}` : '—')
  const installedChannel = status?.installedChannel ?? (status?.installedTag?.startsWith('main@') ? 'main' : 'stable')
  const sourceReleaseRef = status?.currentVersion ? `v${status.currentVersion}` : '—'
  const channelMatches = installedChannel === channel
  const normalizedInstalledRef = channel === 'main' ? installedRef.replace(/^main@/, '') : installedRef.replace(/^v/, '')
  const normalizedLatestRef = channel === 'main' ? status?.latestTag?.replace(/^main@/, '') : status?.latestTag?.replace(/^v/, '')
  const installedRevision = status?.installedRevision?.toLowerCase()
  const latestRevision = status?.latestRevision?.toLowerCase()
  const sameSourceRevision = Boolean(installedRevision && latestRevision && installedRevision.length >= 7 && latestRevision.length >= 7
    && (installedRevision.startsWith(latestRevision) || latestRevision.startsWith(installedRevision)))
  const stableFallback = channel === 'stable' && installedChannel === 'main'
  const sameSourceStableFallback = stableFallback && sameSourceRevision
  const upToDate = (!stableFallback && sameSourceRevision) || Boolean(channelMatches && normalizedInstalledRef && normalizedLatestRef && (channel === 'main'
    ? normalizedInstalledRef.startsWith(normalizedLatestRef) || normalizedLatestRef.startsWith(normalizedInstalledRef)
    : normalizedLatestRef === normalizedInstalledRef))
  const latestDisplay = status?.latestTag ?? (channelMatches ? `${installedRef} (installed)` : 'Unavailable')
  const label = !status?.helperInstalled ? 'Get LABO AI Setup' : upToDate ? 'Up to date' : stableFallback ? 'Switch to Stable' : status.updateAvailable ? 'Install update' : 'Open LABO AI Setup'
  return <article className="desktop-update-settings">
    <RefreshCw size={15} />
    <div>
      <strong>Desktop updates</strong>
      <p>Choose verified releases for normal use, or follow the newest commit when testing LABO AI development.</p>
      <div aria-label="Desktop update channel" className="desktop-update-channel">
        <button aria-pressed={channel === 'stable'} disabled={Boolean(busy)} onClick={() => void selectChannel('stable')} type="button"><ShieldCheck size={14} /><span><strong>Stable</strong><small>Recommended · published releases</small></span></button>
        <button aria-pressed={channel === 'main'} disabled={Boolean(busy)} onClick={() => void selectChannel('main')} type="button"><FlaskConical size={14} /><span><strong>Main</strong><small>Experimental · latest commit</small></span></button>
      </div>
      <dl className="desktop-update-facts">
        <div><dt>Installed</dt><dd>{installedRef}</dd></div>
        <div><dt>Installed channel</dt><dd>{installedChannel === 'main' ? 'Main' : 'Stable'}</dd></div>
        {installedChannel === 'main' && <div><dt>Source release</dt><dd>{sourceReleaseRef}</dd></div>}
        <div><dt>{channel === 'main' ? 'Latest main commit' : 'Latest stable release'}</dt><dd>{latestDisplay}{status?.cachedLatest ? ' · cached' : ''}</dd></div>
      </dl>
      {checked && !error && !status?.error && <div className={`desktop-update-result ${status?.updateAvailable ? 'update-ready' : upToDate ? 'up-to-date' : 'update-unknown'}`}>
        {upToDate && <CheckCircle2 size={12} />}
        <span>{stableFallback ? sameSourceStableFallback ? `The ${status?.latestTag} source is already installed from Main. Switch only to return to the verified Stable channel.` : `Switch from experimental ${installedRef} to verified Stable ${status?.latestTag}.` : status?.updateAvailable ? `${status.latestTag} is ready to install.` : upToDate ? sameSourceRevision && !channelMatches ? `${status?.latestTag} is the same source revision already installed.` : `You are up to date on ${installedRef}.` : 'Update check completed; the latest revision could not be confirmed.'}</span>
      </div>}
      {(error || status?.error) && <small>{error || status?.error}</small>}
      <div className="desktop-update-actions">
        <button disabled={Boolean(busy)} onClick={check} type="button"><RefreshCw size={12} />{busy === 'check' ? 'Checking…' : 'Check for updates'}</button>
        <button disabled={Boolean(busy) || !status || upToDate} onClick={start} type="button">{upToDate ? <CheckCircle2 size={12} /> : <Download size={12} />}{busy === 'launch' ? 'Opening…' : label}</button>
      </div>
    </div>
  </article>
}
