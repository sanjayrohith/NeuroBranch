import { createElement } from './dom.js'

export function createHeroSection() {
  return createElement('section', { className: 'setup-hero' }, [
    createElement('span', { className: 'eyebrow', text: 'SOURCE-FIRST DESKTOP' }),
    createElement('h2', { text: 'Install the latest LABO AI locally.' }),
    createElement('p', {
      text: 'The setup fetches the newest published GitHub release, prepares its own verified Node.js runtime, and builds the Electron laboratory on this computer.',
    }),
  ])
}
