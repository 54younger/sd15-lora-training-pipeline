import { describe, expect, it } from "vitest";
import { ApiError } from "./api";
import type { Job } from "./types";
import {
  formatError,
  jobError,
  phaseLabel,
  progressModel,
  stageModel,
} from "./workflow";

const job = (patch: Partial<Job> = {}): Job => ({
  id: "job-1",
  dataset_id: "dataset-1",
  state: "RUNNING",
  current_stage: "TRAIN",
  stages: {},
  ...patch,
});

describe("workflow stage model", () => {
  it("combines captioning into prepare and exposes its phase", () => {
    const model = stageModel(
      job({
        current_stage: "CAPTION",
        stages: { CAPTION: { state: "RUNNING", attempt_count: 2 } },
      }),
      "PREPARE",
    );
    expect(model.status).toBe("running");
    expect(model.phase).toBe("Generating captions");
    expect(model.attemptCount).toBe(2);
  });

  it("maps queued, waiting, terminal, failure, and cancellation states", () => {
    expect(
      stageModel(
        job({ state: "ACCEPTED", stages: { TRAIN: { state: "PENDING" } } }),
        "TRAIN",
      ).status,
    ).toBe("queued");
    expect(
      stageModel(job({ state: "WAITING_FOR_USER", waiting_for_stage: "EVALUATE" }), "EVALUATE").status,
    ).toBe("waiting");
    expect(stageModel(job({ state: "READY", current_stage: "PUBLISH" }), "PUBLISH").status).toBe("completed");
    expect(stageModel(job({ state: "FAILED", current_stage: "TRAIN" }), "TRAIN").status).toBe("failed");
    expect(stageModel(job({ state: "CANCEL_REQUESTED", current_stage: "TRAIN" }), "TRAIN").status).toBe("cancelling");
    expect(stageModel(job({ state: "CANCELLED", current_stage: "TRAIN" }), "TRAIN").status).toBe("cancelled");
  });

  it("clamps numeric progress and never invents a percentage", () => {
    expect(progressModel("TRAIN", "running", { current: 110, total: 100 })).toMatchObject({
      kind: "determinate",
      percent: 100,
      current: 110,
      total: 100,
    });
    expect(progressModel("TRAIN", "running", { current: -5, total: 100 })).toMatchObject({
      kind: "determinate",
      percent: 0,
    });
    expect(progressModel("TRAIN", "running", {})).toMatchObject({ kind: "indeterminate" });
    expect(progressModel("TRAIN", "queued", {})).toMatchObject({ kind: "indeterminate" });
    expect(progressModel("TRAIN", "completed", {})).toMatchObject({ kind: "determinate", percent: 100 });
  });

  it("keeps training setup and model/device loading indeterminate", () => {
    for (const phase of ["sd15_unet_loading", "cuda_memory_initializing"]) {
      const progress = progressModel("TRAIN", "running", {
        phase,
        current: 0,
        total: 100,
      });
      expect(progress).toMatchObject({ kind: "indeterminate" });
      expect(progress).not.toHaveProperty("percent");
      expect(progress.label).not.toContain("0 / 100");
    }
    expect(
      progressModel("TRAIN", "running", {
        phase: "training",
        current: 10,
        total: 100,
      }),
    ).toMatchObject({
      kind: "determinate",
      percent: 10,
      current: 10,
      total: 100,
    });
  });

  it("models training numbers, evaluation work, and a failed attempt", () => {
    const training = stageModel(
      job({
        stages: {
          TRAIN: {
            state: "RUNNING",
            attempt_count: 1,
            progress: { current: 10, total: 100, phase: "training" },
          },
        },
      }),
      "TRAIN",
    );
    expect(training.progress).toMatchObject({
      kind: "determinate",
      percent: 10,
      current: 10,
      total: 100,
    });
    expect(training.phase).toBe("Training");
    const evaluation = stageModel(
      job({ current_stage: "EVALUATE", stages: { EVALUATE: { state: "RUNNING" } } }),
      "EVALUATE",
    );
    expect(evaluation.progress).toMatchObject({ kind: "indeterminate" });
    const failed = jobError(
      job({
        state: "FAILED",
        current_stage: "TRAIN",
        error_code: "OOM",
        error_message: "CUDA ran out of memory",
        stages: { TRAIN: { state: "FAILED", attempt_count: 3 } },
      }),
      "TRAIN",
    );
    expect(failed).toMatchObject({ code: "OOM", message: "CUDA ran out of memory" });
    const detailed = jobError(
      job({
        state: "FAILED",
        error_code: "GPU_DEVICE_INITIALIZATION_FAILED",
        error_message: "Could not initialize the selected CUDA device",
        error_details: { exception_type: "RuntimeError" },
      }),
      "TRAIN",
    );
    expect(detailed?.details).toEqual({ exception_type: "RuntimeError" });
    expect(["PREPARE", "TRAIN", "EVALUATE", "PUBLISH"].map((stage) =>
      stageModel(job(), stage as "PREPARE" | "TRAIN" | "EVALUATE" | "PUBLISH").label,
    )).toEqual(["Prepare Dataset", "Train Model", "Evaluate", "Publish & Download"]);
  });

  it("keeps evaluation sub-operations indeterminate as their local counters reset", () => {
    for (const progress of [
      { phase: "technical_smoke_completed", current: 1, total: 1 },
      { phase: "baseline_generation_model_loading", current: 0, total: 8 },
      { phase: "baseline_generation", current: 8, total: 8 },
      { phase: "adapter_generation", current: 0, total: 8 },
      { phase: "adapter_clip_scoring", current: 2, total: 20 },
      { phase: "baseline_clip_scoring_training_references", current: 0, total: 20 },
    ]) {
      const model = stageModel(
        job({
          current_stage: "EVALUATE",
          stages: { EVALUATE: { state: "RUNNING", progress } },
        }),
        "EVALUATE",
      );
      expect(model.progress).toMatchObject({ kind: "indeterminate" });
      expect(model.progress.label).toContain("in current operation");
    }
    expect(phaseLabel("baseline_generation_model_loading", "EVALUATE")).toBe(
      "Loading model for baseline generation",
    );
    expect(phaseLabel("adapter_clip_scoring_training_references", "EVALUATE")).toBe(
      "Scoring training reference images for adapter",
    );
  });

  it("formats common phases and structured API errors", () => {
    expect(phaseLabel("technical_smoke_test", "EVALUATE")).toBe(
      "Running technical smoke test",
    );
    expect(phaseLabel("checkpoint_loading", "TRAIN")).toBe("Loading checkpoint");
    expect(
      formatError(
        new ApiError(429, "RATE_LIMITED", "Try again later", { limit: 1 }, "req-1", true),
      ),
    ).toEqual({
      message: "Try again later",
      code: "RATE_LIMITED",
      httpStatus: 429,
      retryable: true,
      requestId: "req-1",
      details: { limit: 1 },
    });
  });
});
