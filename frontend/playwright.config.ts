import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 240_000,
  retries: 0,
  workers: 1,
  expect: { timeout: 15_000 },
  use: {
    baseURL: process.env.STUDIO_E2E_BASE_URL || "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    viewport: { width: 1440, height: 1000 },
    actionTimeout: 20_000,
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    },
  },
});
