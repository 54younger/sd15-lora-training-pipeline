import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./api";
import type { Artifact, Job } from "./types";
import s from "./App.module.css";

const idOf = (job: Job) => job.id || job.job_id || "";
const message = (error: unknown) =>
  error instanceof Error ? error.message : "Unable to load this artifact.";
const pageSize = 20;

function useArtifacts(job: Job) {
  return useQuery({
    queryKey: [
      "artifacts",
      idOf(job),
      Boolean(job.input),
      Boolean(job.training),
      Boolean(job.evaluation),
    ],
    queryFn: () => apiClient.artifacts(idOf(job)),
  });
}

export function Images({ job, kind }: { job: Job; kind: string }) {
  const query = useArtifacts(job);
  const [page, setPage] = useState(0);
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const all = useMemo(
    () => query.data?.artifacts.filter((item) => item.kind === kind) || [],
    [query.data, kind],
  );
  const visible = useMemo(
    () => all.slice(page * pageSize, (page + 1) * pageSize),
    [all, page],
  );
  useEffect(() => setPage(0), [job.id, kind]);
  useEffect(() => {
    let alive = true;
    let cursor = 0;
    const created: string[] = [];
    setUrls({});
    setError("");
    // Fetch only the visible page, with a bounded number of requests in flight.
    const load = async () => {
      while (alive && cursor < visible.length) {
        const item = visible[cursor++];
        try {
          const blob = await apiClient.blob(item.url);
          if (!alive) return;
          const url = URL.createObjectURL(blob);
          created.push(url);
          setUrls((previous) => ({ ...previous, [item.id]: url }));
        } catch (reason) {
          if (alive) setError(message(reason));
        }
      }
    };
    void Promise.all(Array.from({ length: Math.min(4, visible.length) }, load));
    return () => {
      alive = false;
      created.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [visible]);
  if (query.isLoading) return <p className={s.loading}>Loading artifacts…</p>;
  if (query.isError)
    return (
      <p className={s.error} role="alert">
        {message(query.error)}
      </p>
    );
  return (
    <>
      {error && (
        <p className={s.error} role="alert">
          {error}
        </p>
      )}
      {!all.length && (
        <p className={s.muted}>
          {job.profile?.config.backend === "tiny" && kind === "evaluation_image"
            ? "Demo / Test Only — generated images and semantic quality metrics are unavailable for this CPU backend."
            : "No image artifacts are available for this stage yet."}
        </p>
      )}
      <div className={s.artifactGrid}>
        {visible.map((item) => (
          <figure className={s.artifact} key={item.id}>
            {urls[item.id] ? (
              <img src={urls[item.id]} alt={item.label} />
            ) : (
              <p className={s.loading}>Loading image…</p>
            )}
            <figcaption>
              <strong>
                {item.variant === "adapter"
                  ? "LoRA adapter"
                  : item.variant === "base"
                    ? "Base model"
                    : item.name || item.label}
              </strong>
              {item.split && (
                <div>
                  {item.split === "train"
                    ? "Training split"
                    : "Validation split"}
                </div>
              )}
              {item.caption && <div>Caption: {item.caption}</div>}
              {item.prompt && <div>{item.prompt}</div>}
              {item.seed !== undefined && <div>Seed {item.seed}</div>}
            </figcaption>
          </figure>
        ))}
      </div>
      {all.length > pageSize && (
        <div className={s.actions}>
          <button
            className={`${s.button} ${s.buttonSecondary}`}
            disabled={page === 0}
            onClick={() => setPage((value) => value - 1)}
          >
            Previous images
          </button>
          <span className={s.muted}>
            {page * pageSize + 1}–{Math.min((page + 1) * pageSize, all.length)}{" "}
            of {all.length}
          </span>
          <button
            className={`${s.button} ${s.buttonSecondary}`}
            disabled={(page + 1) * pageSize >= all.length}
            onClick={() => setPage((value) => value + 1)}
          >
            Next images
          </button>
        </div>
      )}
      <ArtifactDownloads
        job={job}
        kinds={
          kind === "prepared_image"
            ? ["prepared_manifest", "input_manifest"]
            : ["evaluation_report"]
        }
      />
    </>
  );
}

function ArtifactDownload({ artifact }: { artifact: Artifact }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const alive = useRef(false);
  const currentUrl = useRef("");
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (currentUrl.current) URL.revokeObjectURL(currentUrl.current);
    };
  }, []);
  const prepare = async () => {
    setBusy(true);
    setError("");
    try {
      const blob = await apiClient.blob(artifact.url);
      if (!alive.current) return;
      const next = URL.createObjectURL(blob);
      if (currentUrl.current) URL.revokeObjectURL(currentUrl.current);
      currentUrl.current = next;
      setUrl(next);
    } catch (reason) {
      if (alive.current) setError(message(reason));
    } finally {
      if (alive.current) setBusy(false);
    }
  };
  return (
    <div>
      {url ? (
        <a className={s.download} href={url} download={`${artifact.kind}.json`}>
          Download {artifact.label}
        </a>
      ) : (
        <button
          className={`${s.button} ${s.buttonSecondary}`}
          disabled={busy}
          onClick={() => void prepare()}
        >
          {busy ? "Preparing…" : `Prepare ${artifact.label}`}
        </button>
      )}
      {error && (
        <p className={s.error} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export function ArtifactDownloads({
  job,
  kinds,
}: {
  job: Job;
  kinds: string[];
}) {
  const query = useArtifacts(job);
  if (query.isError) return <p className={s.error}>{message(query.error)}</p>;
  return (
    <div className={s.actions} style={{ flexWrap: "wrap" }}>
      {query.data?.artifacts
        .filter((item) => kinds.includes(item.kind))
        .map((item) => (
          <ArtifactDownload key={item.id} artifact={item} />
        ))}
    </div>
  );
}

export function PreparedStats({ job }: { job: Job }) {
  const statistics = job.prepared?.statistics as
    Record<string, number> | undefined;
  const fields = [
    ["accepted_unique", "Accepted unique"],
    ["train_images", "Training images"],
    ["validation_images", "Validation images"],
    ["exact_duplicates", "Duplicates removed"],
    ["rejected", "Rejected images"],
  ] as const;
  return (
    <div className={s.metrics}>
      {fields.map(([key, label]) => (
        <div className={s.metric} key={key}>
          <span className={s.muted}>{label}</span>
          <b>{statistics?.[key] ?? "—"}</b>
        </div>
      ))}
    </div>
  );
}
