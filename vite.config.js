import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',   // 👈 Listen on all IPs, not just localhost
    port: 5173,        // 👈 Optional: default port, change if needed
    strictPort: true,  // 👈 Don't fallback to another port
  },
});