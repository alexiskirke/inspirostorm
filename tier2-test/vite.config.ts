import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Standalone testbench. Lives on a different port than the main app
// (5174 vs 5173) so both can be running side-by-side without clash.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': 'http://localhost:3002',
    },
  },
});
