import { createElement } from './dom.js'

export function createBrandHeader() {
  const mark = createElement('div', { className: 'brand-mark', attributes: { 'aria-hidden': 'true' } }, [
    createElement('i'),
    createElement('i'),
    createElement('i'),
    createElement('i'),
  ])
  const identity = createElement('div', { className: 'brand-header__identity' }, [
    createElement('span', { text: 'COMPLEXITY' }),
    createElement('h1', { text: 'LABO AI Setup' }),
  ])
  return createElement('header', { className: 'brand-header' }, [mark, identity])
}
