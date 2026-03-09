import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: process.env.LABO_WEB_BASE || './',
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.endsWith('/src/model/AskLaboPanel.tsx')) return 'ask-labo'
        },
      },
    },
  },
})
