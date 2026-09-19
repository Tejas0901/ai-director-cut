import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy keeps the browser on one origin, so uploads, SSE and the <video>
// sources all share a host and we never touch CORS in the happy path.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/media": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
