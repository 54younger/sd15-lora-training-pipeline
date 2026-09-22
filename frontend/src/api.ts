import type { ApiList, Artifact, Dataset, Health, Job, Profile } from "./types";

const BASE = "/api";
const activeControllers = new Set<AbortController>();

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown,
    public requestId?: string,
    public retryable?: boolean,
  ) {
    super(message);
  }
}

export const token = () => sessionStorage.getItem("lora-token") || "";
export function abortPendingRequests() {
  activeControllers.forEach((controller) => controller.abort());
  activeControllers.clear();
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object")
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, entry]) => `${JSON.stringify(key)}:${canonical(entry)}`)
      .join(",")}}`;
  return JSON.stringify(value);
}
export const requestId = (scope: string, payload?: unknown) => {
  const key = `lora-idem:${scope}:${payload === undefined ? "" : canonical(payload)}`;
  const old = sessionStorage.getItem(key);
  if (old) return old;
  const next = crypto.randomUUID();
  sessionStorage.setItem(key, next);
  return next;
};
export const clearId = (scope: string, payload?: unknown) =>
  sessionStorage.removeItem(
    `lora-idem:${scope}:${payload === undefined ? "" : canonical(payload)}`,
  );
async function decode(response: Response) {
  if (response.status === 204) return undefined;
  const contentType = response.headers.get("content-type") || "";
  const data: unknown = contentType.includes("json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const body = (data && typeof data === "object" ? data : {}) as {
      error?: {
        code?: string;
        message?: string;
        details?: unknown;
        request_id?: string;
        retryable?: boolean;
      };
      detail?: string | Array<{ msg?: string }>;
    };
    const detail = Array.isArray(body.detail)
      ? body.detail
          .map((d) => d.msg)
          .filter(Boolean)
          .join("; ")
      : body.detail;
    throw new ApiError(
      response.status,
      body.error?.code || `HTTP_${response.status}`,
      body.error?.message || detail || `Request returned ${response.status}.`,
      body.error?.details ?? data,
      body.error?.request_id ||
        response.headers.get("X-Request-ID") ||
        undefined,
      body.error?.retryable,
    );
  }
  return data;
}
export async function api<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  activeControllers.add(controller);
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${token()}`);
  if (
    init.body &&
    !(init.body instanceof Blob) &&
    !(init.body instanceof ArrayBuffer)
  )
    headers.set("Content-Type", "application/json");
  const abort = () => controller.abort();
  if (signal?.aborted) controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  try {
    const response = await fetch(`${BASE}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
    return (await decode(response)) as T;
  } finally {
    activeControllers.delete(controller);
    signal?.removeEventListener("abort", abort);
  }
}
async function blob(path: string): Promise<Blob> {
  const controller = new AbortController();
  activeControllers.add(controller);
  try {
    const response = await fetch(BASE + path, {
      headers: { Authorization: "Bearer " + token() },
      signal: controller.signal,
    });
    if (!response.ok) await decode(response);
    return await response.blob();
  } finally {
    activeControllers.delete(controller);
  }
}
export const mutation = <T>(
  path: string,
  body: unknown,
  scope: string,
  method = "POST",
) =>
  api<T>(path, {
    method,
    body: JSON.stringify(body),
    headers: { "Idempotency-Key": requestId(scope, body) },
  });
export const apiClient = {
  profiles: () => api<{ profiles: Profile[] }>("/v1/training-profiles"),
  datasets: (limit = 50, offset = 0) =>
    api<ApiList<Dataset>>(`/v1/datasets?limit=${limit}&offset=${offset}`),
  dataset: (id: string) =>
    api<Dataset>(`/v1/datasets/${id}?include_files=true`),
  jobs: (state?: string, limit = 50, offset = 0) =>
    api<ApiList<Job>>(
      `/v1/training-jobs?limit=${limit}&offset=${offset}${state ? `&state=${encodeURIComponent(state)}` : ""}`,
    ),
  job: (id: string, summary = true) =>
    api<Job>(`/v1/training-jobs/${id}${summary ? "?view=summary" : ""}`),
  evaluation: (id: string) =>
    api<Record<string, unknown>>(`/v1/training-jobs/${id}/evaluation`),
  artifacts: (id: string) =>
    api<{ artifacts: Artifact[] }>(`/v1/training-jobs/${id}/artifacts`),
  ready: async (): Promise<Health> => {
    try {
      return await api<Health>("/health/ready");
    } catch (error) {
      if (
        error instanceof ApiError &&
        error.status === 503 &&
        error.details &&
        typeof error.details === "object"
      )
        return error.details as Health;
      throw error;
    }
  },
  createDataset: (files: unknown[], intent = "dataset") =>
    mutation<Dataset>("/v1/datasets", { files }, intent),
  completeDataset: (id: string) =>
    mutation<Dataset>(`/v1/datasets/${id}/complete`, {}, `complete:${id}`),
  createJob: (body: unknown, intent = "job") =>
    mutation<Job>("/v1/training-jobs", body, intent),
  advance: (id: string, body: unknown) =>
    mutation<Job>(`/v1/training-jobs/${id}/advance`, body, `advance:${id}`),
  cancel: (id: string) =>
    mutation<Job>(`/v1/training-jobs/${id}/cancel`, {}, `cancel:${id}`),
  blob,
};
export function upload(
  datasetId: string,
  fileId: string,
  body: File,
  onProgress: (n: number) => void,
  signal?: AbortSignal,
) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Upload cancelled", "AbortError"));
      return;
    }
    const xhr = new XMLHttpRequest();
    const controller = new AbortController();
    activeControllers.add(controller);
    xhr.open("PUT", `${BASE}/v1/datasets/${datasetId}/files/${fileId}`);
    xhr.setRequestHeader("Authorization", `Bearer ${token()}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded);
    };
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(
            new ApiError(
              xhr.status,
              `HTTP_${xhr.status}`,
              xhr.responseText || "Upload failed.",
            ),
          );
    xhr.onerror = () => reject(new Error("Upload network error."));
    xhr.onabort = () =>
      reject(new DOMException("Upload cancelled", "AbortError"));
    const abort = () => xhr.abort();
    controller.signal.addEventListener("abort", abort, { once: true });
    signal?.addEventListener("abort", abort, { once: true });
    xhr.onloadend = () => {
      activeControllers.delete(controller);
      signal?.removeEventListener("abort", abort);
    };
    xhr.send(body);
  });
}
