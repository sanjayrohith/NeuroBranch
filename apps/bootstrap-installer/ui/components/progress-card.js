import { createElement } from './dom.js'

export function createProgressCard() {
  const stage = createElement('span', { className: 'progress-card__stage', text: 'Ready' })
  const version = createElement('strong', { className: 'progress-card__version', text: 'Checking latest release...' })
  const fill = createElement('i', { className: 'progress-card__fill' })
  const message = createElement('p', {
    className: 'progress-card__message',
    text: 'Your workspaces and private cards stay in your local LABO AI profile during updates.',
  })
  const log = createElement('ol', { className: 'progress-card__log' })
  const details = createElement('details', { className: 'progress-card__details', attributes: { open: '' } }, [
    createElement('summary', { text: 'Installation details' }),
    log,
  ])
  const element = createElement('section', { className: 'progress-card' }, [
    createElement('div', { className: 'progress-card__header' }, [stage, version]),
    createElement('div', { className: 'progress-card__track', attributes: { 'aria-hidden': 'true' } }, [fill]),
    message,
    details,
  ])

  function appendLog(step, text) {
    const item = createElement('li', {}, [
      createElement('time', { text: step }),
      createElement('span', { text }),
    ])
    log.append(item)
    log.scrollTop = log.scrollHeight
  }

  appendLog('Ready', 'Waiting to start.')

  return {
    element,
    appendLog,
    setMessage(value) { message.textContent = value },
    setVersion(value) { version.textContent = value },
    update(payload) {
      if (payload.stage !== undefined) stage.textContent = payload.stage
      if (payload.message !== undefined) message.textContent = payload.message
      if (payload.percent !== undefined) fill.style.width = `${Math.max(0, Math.min(100, payload.percent))}%`
      details.open = true
    },
  }
}
