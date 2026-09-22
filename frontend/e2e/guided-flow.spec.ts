import { expect, test } from "@playwright/test";
import path from "node:path";
import { readdir } from "node:fs/promises";

const token = process.env.STUDIO_E2E_TOKEN;
const imageDir = process.env.STUDIO_E2E_IMAGES_DIR;
const captions = process.env.STUDIO_E2E_CAPTIONS_PATH;

test.skip(
  !token || !imageDir || !captions,
  "Run with scripts/web_e2e_fixture.py environment variables.",
);
test("complete all four stages against the real API and worker", async ({
  page,
  request,
}, testInfo) => {
  const headers = { Authorization: `Bearer ${token}` };
  const images = (await readdir(imageDir!))
    .filter((name) => name.endsWith(".png"))
    .sort();
  expect(images.length).toBeGreaterThanOrEqual(100);
  await page.goto("/");
  await page.getByLabel("API token").fill(token!);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("link", { name: "New Training" }).click();
  await page.getByLabel("Trigger token").fill("studiofixture");
  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles([
      ...images.map((name) => path.join(imageDir!, name)),
      captions!,
    ]);
  await expect(
    page.getByLabel(`Caption for ${images[0]}`, { exact: true }),
  ).not.toHaveValue("");
  await page
    .getByLabel(`Caption for ${images[0]}`, { exact: true })
    .fill("studiofixture edited product image");
  await page.getByRole("button", { name: "Process dataset" }).click();
  await expect(
    page.getByRole("heading", { name: "Confirm training setup" }),
  ).toBeVisible({ timeout: 90_000 });
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Confirm training setup" }),
  ).toBeVisible();
  const id = new URL(page.url()).pathname.split("/").pop()!;
  const endpoint = `/api/v1/training-jobs/${id}`;
  const read = async () => {
    const response = await request.get(endpoint, { headers });
    expect(response.ok()).toBeTruthy();
    return response.json();
  };
  const waiting = async (stage: string) => {
    await expect
      .poll(
        async () => {
          const job = await read();
          return `${job.state}:${job.waiting_for_stage}`;
        },
        { timeout: 90_000 },
      )
      .toBe(`WAITING_FOR_USER:${stage}`);
    await page.reload();
    const job = await read();
    expect(job.stages[stage]).toBeUndefined();
    expect((await read()).waiting_for_stage).toBe(stage);
    return job;
  };
  const prepared = await waiting("TRAIN");
  expect(
    [...prepared.input.train, ...prepared.prepared.validation].some(
      (item: { caption?: string }) => item.caption?.includes("edited product"),
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: /Prepare Dataset/ }).click();
  await expect(
    page.getByText("Accepted unique", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("img").first()).toHaveAttribute("src", /^blob:/);
  await page.getByRole("button", { name: "Next images", exact: true }).click();
  await expect(page.getByText("21–40 of 100", { exact: true })).toBeVisible();
  await page
    .getByRole("button", {
      name: "Prepare Prepared dataset manifest",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("link", {
      name: "Download Prepared dataset manifest",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Prepare Training input manifest",
      exact: true,
    })
    .click();
  const manifestDownloadEvent = page.waitForEvent("download");
  await page
    .getByRole("link", {
      name: "Download Prepared dataset manifest",
      exact: true,
    })
    .click();
  expect(await (await manifestDownloadEvent).failure()).toBeNull();
  await page.getByRole("button", { name: /Train Model/ }).click();
  await page.screenshot({
    path: testInfo.outputPath("prepared.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Start training", exact: true })
    .click();
  await waiting("EVALUATE");
  await page
    .getByRole("button", { name: "Run evaluation", exact: true })
    .click();
  const evaluated = await waiting("PUBLISH");
  expect(evaluated.evaluation.technical_pass).toBe(true);
  expect(evaluated.evaluation.test_only).toBe(true);
  await page.getByRole("button", { name: /^3\.\s*Evaluate$/ }).click();
  await expect(page.getByText("Passed", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Prepare Evaluation report", exact: true })
    .click();
  const reportDownloadEvent = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "Download Evaluation report", exact: true })
    .click();
  expect(await (await reportDownloadEvent).failure()).toBeNull();
  await page.getByRole("button", { name: /Publish & Download/ }).click();
  await page
    .getByRole("button", { name: "Publish model", exact: true })
    .click();
  await expect
    .poll(async () => (await read()).state)
    .toBe("COMPLETED_UNVERIFIED");
  await page.reload();
  const result = await read();
  expect(
    (
      await request.get(`/api/v1/models/${result.model_id}/download`, {
        headers,
      })
    ).status(),
  ).toBe(409);
  await page.getByRole("checkbox").check();
  await page
    .getByRole("button", { name: "Prepare adapter download", exact: true })
    .click();
  const event = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "Download adapter", exact: true })
    .click();
  const download = await event;
  expect(download.suggestedFilename()).toMatch(/\.safetensors$/);
  expect(await download.failure()).toBeNull();
  await page.screenshot({
    path: testInfo.outputPath("published.png"),
    fullPage: true,
  });
  const otherHeaders = { Authorization: "Bearer other-test-token" };
  expect(
    (await request.get(endpoint, { headers: otherHeaders })).status(),
  ).toBe(404);
  const artifacts = await request.get(`${endpoint}/artifacts`, { headers });
  const artifact = (await artifacts.json()).artifacts[0];
  expect(
    (
      await request.get(`/api${artifact.url}`, { headers: otherHeaders })
    ).status(),
  ).toBe(404);
  await page.getByRole("link", { name: "Training Runs", exact: true }).click();
  await expect(page.getByText(id.slice(0, 8), { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: testInfo.outputPath("mobile-history.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByLabel("API token")).toBeVisible();
  expect(
    await page.evaluate(() => sessionStorage.getItem("lora-token")),
  ).toBeNull();
});

test("a rejected upload releases the form for corrected data", async ({
  page,
}) => {
  await page.addInitScript(
    ({ token }) => {
      sessionStorage.setItem("lora-token", token!);
      sessionStorage.setItem(
        "lora-pending-upload",
        JSON.stringify({
          id: "rejected-fixture",
          profileId: "local-sd15-v1",
          trigger: "rejected",
          intent: "rejected-intent",
          files: [],
        }),
      );
    },
    { token },
  );
  await page.route("**/api/v1/datasets/rejected-fixture?*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "rejected-fixture",
        state: "INVALID",
        files: [],
        verification_error: "CHECKSUM_MISMATCH",
      }),
    }),
  );
  await page.goto("/new");
  await expect(
    page.getByText(/Dataset validation rejected this upload/),
  ).toBeVisible();
  expect(
    await page.evaluate(() => sessionStorage.getItem("lora-pending-upload")),
  ).toBeNull();
  await expect(page.getByLabel("Trigger token", { exact: true })).toBeEnabled();
  await expect(page.locator("input[type=file]").first()).toBeEnabled();
});

test("discarding upload recovery unlocks a fresh training draft", async ({
  page,
}) => {
  let releaseDataset!: () => void;
  let markDatasetReturned!: () => void;
  const datasetStarted = new Promise<void>((resolve) => {
    releaseDataset = resolve;
  });
  const datasetReturned = new Promise<void>((resolve) => {
    markDatasetReturned = resolve;
  });
  await page.addInitScript(
    ({ token }) => {
      sessionStorage.setItem("lora-token", token!);
      sessionStorage.setItem(
        "lora-pending-upload",
        JSON.stringify({
          id: "discard-fixture",
          profileId: "local-sd15-v1",
          trigger: "old-trigger",
          intent: "old-intent",
          files: [],
        }),
      );
      sessionStorage.setItem("lora-draft-intent", "old-intent");
    },
    { token },
  );
  await page.route("**/api/v1/datasets/discard-fixture?*", async (route) => {
    await datasetStarted;
    try {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "discard-fixture",
          state: "INVALID",
          verification_error: "OLD_UPLOAD_SHOULD_NOT_APPEAR",
          files: [
            {
              id: "old-file",
              name: "stale-upload.png",
              size_bytes: 1,
              sha256: "old-hash",
              mime_type: "image/png",
              uploaded: false,
            },
          ],
        }),
      });
    } finally {
      markDatasetReturned();
    }
  });
  await page.setViewportSize({ width: 390, height: 844 });
  const pendingDatasetRequest = page.waitForRequest(
    "**/api/v1/datasets/discard-fixture?*",
  );
  await page.goto("/new");
  await pendingDatasetRequest;
  await expect(page.getByTestId("upload-recovery")).toBeVisible();
  const recoveryActions = page.getByTestId("upload-recovery-actions");
  await expect(recoveryActions.locator(":scope > *")).toHaveCount(3);
  const actionBoxes = await recoveryActions.locator(":scope > *").evaluateAll(
    (elements) =>
      elements.map((element) => {
        const { width, y } = element.getBoundingClientRect();
        return { width, y };
      }),
  );
  expect(actionBoxes[0].y).toBeLessThan(actionBoxes[1].y);
  expect(actionBoxes[1].y).toBeLessThan(actionBoxes[2].y);
  actionBoxes.forEach((box) =>
    expect(Math.abs(box.width - actionBoxes[0].width)).toBeLessThanOrEqual(2),
  );
  await page
    .getByRole("button", { name: "Discard upload and start new", exact: true })
    .click();
  releaseDataset();
  await datasetReturned;
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  await expect(page.getByTestId("upload-recovery")).not.toBeVisible();
  await expect(page).toHaveURL(/\/new$/);
  await expect(page.getByLabel("Trigger token", { exact: true })).toBeEnabled();
  await expect(page.getByLabel("Trigger token", { exact: true })).toHaveValue("");
  await expect(page.locator("input[type=file]").first()).toBeEnabled();
  await expect(page.getByText("OLD_UPLOAD_SHOULD_NOT_APPEAR")).not.toBeVisible();
  await expect(page.getByText("stale-upload.png")).not.toBeVisible();
  expect(
    await page.evaluate(() => sessionStorage.getItem("lora-pending-upload")),
  ).toBeNull();
  expect(
    await page.evaluate(() => sessionStorage.getItem("lora-draft-intent")),
  ).not.toBe("old-intent");
});

test("resume a failed upload after refreshing without duplicating the dataset", async ({
  page,
  request,
}) => {
  const headers = { Authorization: `Bearer ${token}` };
  const before = await (
    await request.get("/api/v1/datasets", { headers })
  ).json();
  const images = (await readdir(imageDir!))
    .filter((name) => name.endsWith(".png"))
    .sort();
  await page.goto("/");
  await page.getByLabel("API token").fill(token!);
  await page.getByRole("button", { name: "Connect", exact: true }).click();
  await page.getByRole("link", { name: "New Training", exact: true }).click();
  await page
    .getByLabel("Trigger token", { exact: true })
    .fill("uploadrecovery");
  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles([
      ...images.map((name) => path.join(imageDir!, name)),
      captions!,
    ]);
  let failed = false;
  await page.route("**/api/v1/datasets/*/files/*", async (route) => {
    if (!failed) {
      failed = true;
      await route.abort("failed");
    } else await route.continue();
  });
  await page
    .getByRole("button", { name: "Process dataset", exact: true })
    .click();
  await expect(
    page.getByText("Upload network error.", { exact: true }),
  ).toBeVisible();
  await page.unroute("**/api/v1/datasets/*/files/*");
  await page.reload();
  await expect(page.getByTestId("upload-recovery")).toBeVisible();
  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles(images.map((name) => path.join(imageDir!, name)));
  await expect(
    page.getByRole("heading", { name: "Confirm training setup", exact: true }),
  ).toBeVisible({ timeout: 90_000 });
  const after = await (
    await request.get("/api/v1/datasets", { headers })
  ).json();
  expect(after.total).toBe(before.total + 1);
  await page.getByRole("button", { name: "Cancel run", exact: true }).click();
  await expect(page.getByText("cancelled", { exact: true })).toBeVisible();
});
