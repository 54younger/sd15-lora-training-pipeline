import { ApiError } from "./api";
import type { Job, Stage, StageTask } from "./types";

export type StageStatus =
  | "pending"
  | "queued"
  | "running"
  | "waiting"
  | "completed"
  | "failed"
  | "cancelling"
  | "cancelled";
export type ProgressState =
  | { kind: "determinate"; percent: number; label: string; current: number; total: number }
  | { kind: "indeterminate"; label: string }
  | { kind: "static"; label: string };
export type StageModel = {
  stage: Stage;
  label: string;
  status: StageStatus;
  statusLabel: string;
  description: string;
  phase: string;
  attemptCount: number;
  progress: ProgressState;
  task?: StageTask;
};
export type ErrorModel = {
  message: string;
  code?: string;
  httpStatus?: number;
  retryable?: boolean;
  requestId?: string;
  details?: unknown;
};

export const stageLabels: Record<Stage, string> = {
  PREPARE: "Prepare Dataset",
  TRAIN: "Train Model",
  EVALUATE: "Evaluate",
  PUBLISH: "Publish & Download",
};
const order: Stage[] = ["PREPARE", "TRAIN", "EVALUATE", "PUBLISH"];
const successfulJobs = new Set(["READY", "COMPLETED_UNVERIFIED"]);

export function stageFromRaw(value?: string | null): Stage {
  return value === "CAPTION" ? "PREPARE" :
    value === "TRAIN" || value === "EVALUATE" || value === "PUBLISH"
      ? value
      : "PREPARE";
}
export function humanize(value?: string | null): string {
  if (!value) return "";
  return value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
export function phaseLabel(phase: unknown, stage: Stage, caption = false): string {
  const raw = typeof phase === "string" ? phase : "";
  const key = raw.toLowerCase().replace(/[-\s]+/g, "_");
  const labels: Record<string, string> = {
    prepare: "Preparing images",
    preparing_images: "Preparing images",
    caption: "Generating captions",
    generating_captions: "Generating captions",
    input_validation: "Validating training input",
    device_resolving: "Selecting training device",
    device_resolved: "Training device ready",
    random_seed_initializing: "Initializing reproducibility",
    random_seed_initialized: "Reproducibility initialized",
    cuda_memory_initializing: "Initializing CUDA memory tracking",
    cuda_memory_initialized: "CUDA memory tracking ready",
    cuda_memory_telemetry_unavailable: "CUDA memory tracking unavailable; training continues",
    model_loading: "Loading model",
    loading_model: "Loading model",
    tiny_components_loading: "Loading test model components",
    sd15_snapshot_resolving: "Resolving SD 1.5 model snapshot",
    sd15_snapshot_resolved: "SD 1.5 model snapshot resolved",
    sd15_tokenizer_loading: "Loading tokenizer",
    sd15_text_encoder_loading: "Loading text encoder",
    sd15_vae_loading: "Loading VAE",
    sd15_unet_loading: "Loading UNet",
    sd15_scheduler_loading: "Loading noise scheduler",
    sd15_adapter_setup: "Configuring LoRA adapter",
    sd15_adapter_ready: "LoRA adapter ready",
    optimizer_initializing: "Initializing optimizer",
    optimizer_initialized: "Optimizer ready",
    checkpoint_loading: "Loading checkpoint",
    loading_checkpoint: "Loading checkpoint",
    checkpoint_restoring: "Restoring checkpoint",
    checkpoint_restored: "Checkpoint restored",
    checkpointing: "Saving checkpoint",
    saving_checkpoint: "Saving checkpoint",
    checkpoint_saving: "Saving checkpoint",
    checkpoint_saved: "Checkpoint saved",
    adapter_saving: "Saving adapter",
    adapter_saved: "Adapter saved",
    report_generation: "Creating report",
    generating_report: "Creating report",
    report_writing: "Writing evaluation report",
    comparison_report_writing: "Writing comparison report",
    technical_smoke: "Running technical smoke test",
    technical_smoke_test: "Running technical smoke test",
    technical_smoke_loading: "Loading technical smoke test",
    technical_smoke_completed: "Technical smoke test completed",
    generation: "Generating evaluation images",
    scoring: "Scoring evaluation results",
    evaluation_started: "Starting evaluation",
    evaluation_completed: "Evaluation completed",
    publish: "Publishing model",
  };
  if (key && labels[key]) return labels[key];
  const generation = key.match(/^(baseline|adapter|left|right)_generation(?:_(model_loading|model_loaded))?$/);
  if (generation) {
    const target = generation[1] === "baseline" ? "baseline" : generation[1] === "adapter" ? "adapter" : generation[1];
    if (generation[2] === "model_loading") return `Loading model for ${target} generation`;
    if (generation[2] === "model_loaded") return `Model loaded for ${target} generation`;
    return `Generating ${target} images`;
  }
  const scoring = key.match(/^(adapter|baseline|left|right)_clip_scoring(?:_(model_loading|heldout_references|training_references))?$/);
  if (scoring) {
    const target = scoring[1];
    if (scoring[2] === "model_loading") return `Loading CLIP scorer for ${target} images`;
    if (scoring[2] === "heldout_references") return `Scoring held-out reference images for ${target}`;
    if (scoring[2] === "training_references") return `Scoring training reference images for ${target}`;
    return `Scoring ${target} images with CLIP`;
  }
  if (key) return humanize(key);
  if (stage === "PREPARE") return caption ? "Generating captions" : "Preparing images";
  if (stage === "TRAIN") return "Training model";
  if (stage === "EVALUATE") return "Evaluating model";
  return "Publishing model";
}

// Training emits the configured max_steps alongside setup events so that every
// progress record has a stable shape. Those numbers are not work completed yet:
// displaying them as 0 / max_steps makes model/device initialization look like
// a stalled training loop. Only the `training` phase represents optimizer steps;
// checkpoint/adapter operations may continue to expose their step counter.
const trainSetupPhases = new Set([
  "training_started",
  "input_validation",
  "device_resolving",
  "device_resolved",
  "random_seed_initializing",
  "random_seed_initialized",
  "cuda_memory_initializing",
  "cuda_memory_initialized",
  "cuda_memory_telemetry_unavailable",
  "model_loading",
  "model_loaded",
  "tiny_components_loading",
  "sd15_snapshot_resolving",
  "sd15_snapshot_resolved",
  "sd15_tokenizer_loading",
  "sd15_text_encoder_loading",
  "sd15_vae_loading",
  "sd15_unet_loading",
  "sd15_scheduler_loading",
  "sd15_adapter_setup",
  "sd15_adapter_ready",
  "optimizer_initializing",
  "optimizer_initialized",
  "checkpoint_restoring",
  "checkpoint_restored",
]);

function progressPhase(value: unknown): string {
  return typeof value === "string" ? value.toLowerCase().replace(/[-\s]+/g, "_") : "";
}

export function progressModel(
  stage: Stage,
  status: StageStatus,
  progress: Record<string, unknown> = {},
): ProgressState {
  if (status === "completed")
    return { kind: "determinate", percent: 100, label: "Complete", current: 1, total: 1 };
  const current = Number(progress.current ?? progress.global_step);
  const total = Number(progress.total ?? progress.total_steps);
  const setupPhase = stage === "TRAIN" && trainSetupPhases.has(progressPhase(progress.phase));
  if (
    stage === "TRAIN" &&
    status === "running" &&
    !setupPhase &&
    Number.isFinite(current) &&
    Number.isFinite(total) &&
    total > 0
  ) {
    const percent = Math.max(0, Math.min(100, (current / total) * 100));
    return {
      kind: "determinate",
      percent,
      label: `${current} / ${total}`,
      current,
      total,
    };
  }
  if (status === "running") {
    const operation = !setupPhase && Number.isFinite(current) && Number.isFinite(total) && total > 0;
    return {
      kind: "indeterminate",
      label: operation
        ? `Working — ${current} / ${total} in current operation`
        : "Working — progress is not reported yet",
    };
  }
  if (status === "queued") return { kind: "indeterminate", label: "Queued — waiting for a worker" };
  return { kind: "static", label: status === "waiting" ? "Waiting for your confirmation" : humanize(status) };
}
function taskFor(job: Job, stage: Stage): { task?: StageTask; caption: boolean } {
  if (stage !== "PREPARE") return { task: job.stages?.[stage], caption: false };
  const caption = job.stages?.CAPTION;
  const useCaption = job.current_stage === "CAPTION" ||
    ["PENDING", "RETRY_WAIT", "RUNNING", "FAILED", "CANCELLED"].includes(caption?.state || "");
  return { task: useCaption ? caption || job.stages?.PREPARE : job.stages?.PREPARE || caption, caption: useCaption };
}
function statusFor(job: Job, stage: Stage, task?: StageTask): StageStatus {
  const current = stageFromRaw(job.current_stage);
  const currentIndex = order.indexOf(current);
  const index = order.indexOf(stage);
  if (job.state === "CANCEL_REQUESTED" && current === stage) return "cancelling";
  if (job.state === "CANCELLED" && current === stage) return "cancelled";
  if (["FAILED", "QUALITY_REJECTED"].includes(job.state) && current === stage) return "failed";
  if (successfulJobs.has(job.state)) return "completed";
  if (job.state === "WAITING_FOR_USER" && job.waiting_for_stage === stage) return "waiting";
  if (task?.state === "SUCCEEDED") return "completed";
  if (task?.state === "FAILED") return "failed";
  if (task?.state === "CANCELLED") return "cancelled";
  if (task?.state === "RUNNING") return "running";
  if (task?.state === "PENDING" || task?.state === "RETRY_WAIT") return "queued";
  if (index < currentIndex) return "completed";
  if (index === currentIndex && ["ACCEPTED", "RUNNING"].includes(job.state)) return job.state === "RUNNING" ? "running" : "queued";
  return "pending";
}
const descriptions: Record<StageStatus, string> = {
  pending: "This stage has not started yet.",
  queued: "The work is queued and will start when capacity is available.",
  running: "Work is in progress. This page refreshes automatically.",
  waiting: "Review the setup and choose when to continue.",
  completed: "This stage completed successfully.",
  failed: "This stage stopped with an error. Review the details below.",
  cancelling: "Cancellation has been requested; the active task is stopping.",
  cancelled: "This stage was cancelled.",
};
export function stageModel(job: Job, stage: Stage): StageModel {
  const { task, caption } = taskFor(job, stage);
  const status = statusFor(job, stage, task);
  const progress = (task?.progress || {}) as Record<string, unknown>;
  return {
    stage,
    label: stageLabels[stage],
    status,
    statusLabel: {
      pending: "Not started",
      queued: task?.state === "RETRY_WAIT" ? "Retry scheduled" : "Queued",
      running: "Running",
      waiting: "Waiting for you",
      completed: "Completed",
      failed: "Failed",
      cancelling: "Cancelling",
      cancelled: "Cancelled",
    }[status],
    description: descriptions[status],
    phase: phaseLabel(progress.phase, stage, caption),
    attemptCount: Number(task?.attempt_count || 0),
    progress: progressModel(stage, status, progress),
    task,
  };
}
export function formatError(error: unknown): ErrorModel {
  if (error instanceof ApiError)
    return {
      message: error.message,
      code: error.code,
      httpStatus: error.status,
      retryable: error.retryable,
      requestId: error.requestId,
      details: error.details,
    };
  if (error instanceof Error) return { message: error.message };
  return { message: "Something went wrong. Please try again." };
}
export function jobError(job: Job, stage: Stage): ErrorModel | undefined {
  const source = job.error;
  const message = source?.message || job.error_message;
  if (!message) return undefined;
  return {
    message,
    code: source?.code || job.error_code,
    retryable: source?.retryable,
    requestId: source?.request_id,
    details: source?.details ?? job.error_details ?? undefined,
  };
}
export function errorDetails(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
