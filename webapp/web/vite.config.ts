import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev proxy target; AURALIS_API=http://127.0.0.1:8766 points the UI at a test server.
const env = (globalThis as unknown as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const API_TARGET = env.AURALIS_API ?? 'http://127.0.0.1:8765';

export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: false },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 5000,
  },
});
