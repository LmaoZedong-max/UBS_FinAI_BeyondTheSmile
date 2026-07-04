import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: process.env.PORT ? Number(process.env.PORT) : 5173,
    allowedHosts: ['.app.github.dev'],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rolldownOptions: {
      output: {
        // Split recharts (+ its d3 deps) and react/react-dom/react-router
        // into separate named chunks to suppress the >500 kB warning.
        // rolldown-vite uses rolldownOptions with codeSplitting.groups.
        codeSplitting: {
          groups: [
            {
              name: 'vendor',
              test: /node_modules[\\/](react|react-dom|react-router|react-router-dom)[\\/]/,
            },
            {
              name: 'charts',
              test: /node_modules[\\/](recharts|d3-|victory-vendor)[\\/]/,
            },
          ],
        },
      },
    },
  },
})
