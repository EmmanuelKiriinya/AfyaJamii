import react from "@vitejs/plugin-react-swc";
import path from "node:path";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: true,
    port: 8080,
    strictPort: true,
  },
  preview: {
    port: 8080,
  },
  build: {
    outDir: "dist",
    // Source maps only outside production, so the shipped bundle does not
    // expose the original sources.
    sourcemap: mode !== "production",
    rollupOptions: {
      output: {
        // Split the vendor code that rarely changes from application code, so
        // a routine deploy does not invalidate the whole cache.
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          radix: [
            "@radix-ui/react-select",
            "@radix-ui/react-tabs",
            "@radix-ui/react-tooltip",
          ],
        },
      },
    },
  },
}));
