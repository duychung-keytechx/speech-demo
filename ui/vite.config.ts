import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE || 'http://192.168.1.8:5000',
        changeOrigin: true,
      },
    },
  },
})
