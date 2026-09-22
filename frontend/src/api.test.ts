import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiClient, requestId } from "./api";

describe("API client", () => {
  beforeEach(() => {
    sessionStorage.clear();
    sessionStorage.setItem("lora-token", "browser-token");
    vi.restoreAllMocks();
  });

  it("uses authenticated dataset detail requests with uploaded files", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(JSON.stringify({ id: "dataset-1", files: [] }), {
          headers: { "content-type": "application/json" },
        }),
      );
    await apiClient.dataset("dataset-1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/datasets/dataset-1?include_files=true",
      expect.objectContaining({ headers: expect.any(Headers) }),
    );
    const headers = fetchMock.mock.calls[0][1]?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer browser-token");
  });

  it("keeps a mutation idempotency key stable through a retry", () => {
    expect(requestId("upload:one")).toBe(requestId("upload:one"));
    expect(requestId("upload:one")).not.toBe(requestId("upload:two"));
  });

  it("normalizes FastAPI validation errors for the UI", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: [{ msg: "Field required" }] }), {
        status: 422,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(apiClient.profiles()).rejects.toMatchObject({
      status: 422,
      message: "Field required",
    } satisfies Partial<ApiError>);
  });

  it("returns binary artifacts intact, including JSON downloads", async () => {
    const bytes = new Uint8Array([0, 255, 1, 128, 42]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(bytes, {
        headers: { "content-type": "application/octet-stream" },
      }),
    );
    const result = await apiClient.blob("/v1/models/example/download");
    expect(new Uint8Array(await result.arrayBuffer())).toEqual(bytes);
  });

  it("shows individual service health when readiness returns 503", async () => {
    const health = {
      status: "degraded",
      database: true,
      storage: true,
      worker: false,
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(health), {
        status: 503,
        headers: { "content-type": "application/json" },
      }),
    );
    expect(await apiClient.ready()).toEqual(health);
  });

  it("retries the same intent but permits an intentional second run", async () => {
    const mock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(
        async () =>
          new Response("{}", {
            headers: { "content-type": "application/json" },
          }),
      );
    const body = { dataset_id: "one", trigger_token: "style" };
    await apiClient.createJob(body, "intent-one");
    await apiClient.createJob(body, "intent-one");
    await apiClient.createJob(body, "intent-two");
    const keys = mock.mock.calls.map((call) =>
      (call[1]?.headers as Headers).get("Idempotency-Key"),
    );
    expect(keys[0]).toBe(keys[1]);
    expect(keys[2]).not.toBe(keys[0]);
  });

  it("changes mutation keys when corrected request data changes", () => {
    expect(
      requestId("advance", {
        stage: "TRAIN",
        training_overrides: { max_steps: 0 },
      }),
    ).not.toBe(
      requestId("advance", {
        stage: "TRAIN",
        training_overrides: { max_steps: 2 },
      }),
    );
  });

  it("preserves diagnostic request IDs for pipeline failures", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            code: "DATASET_INVALID",
            message: "Invalid data",
            request_id: "req-one",
            retryable: false,
          },
        }),
        { status: 422, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(apiClient.profiles()).rejects.toMatchObject({
      code: "DATASET_INVALID",
      requestId: "req-one",
      retryable: false,
    });
  });
});
