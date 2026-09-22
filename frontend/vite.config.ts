import { defineConfig } from "vitest/config";
import { loadEnv as readEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = readEnv(mode, ".", "");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: env.VITE_API_TARGET || "http://localhost:8000",
          rewrite: (p) => p.replace(/^\/api/, ""),
        },
      },
    },
    test: {
      include: ["src/**/*.test.ts"],
      environment: "jsdom",
      globals: true,
    },
  };
});
