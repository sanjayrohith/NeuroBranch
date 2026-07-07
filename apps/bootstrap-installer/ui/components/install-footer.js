import { createElement } from './dom.js'

export function createInstallFooter() {
  const button = createElement('button', { className: 'install-footer__button', text: 'Install latest', attributes: { type: 'button' } })
  const element = createElement('footer', { className: 'install-footer' }, [
    createElement('span', { text: 'Local install · rollback enabled' }),
    button,
  ])
  return { element, button }
}
