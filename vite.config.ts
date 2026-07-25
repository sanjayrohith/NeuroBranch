import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: process.env.NEUROBRANCH_WEB_BASE || './',
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.endsWith('/src/model/AskNeuroBranchPanel.tsx')) return 'ask-neurobranch'
        },
      },
    },
  },
})
