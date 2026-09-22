import { useEffect, useMemo, useRef, useState } from "react";
import {
  NavLink,
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { abortPendingRequests, ApiError, apiClient, upload } from "./api";
import type { Job, Stage } from "./types";
import {
  errorDetails,
  formatError,
  jobError,
  stageLabels,
  stageModel,
  type ErrorModel,
  type StageModel,
} from "./workflow";
import s from "./App.module.css";
import { Images, ArtifactDownloads, PreparedStats } from "./ArtifactViews";

const terminal = [
  "READY",
  "COMPLETED_UNVERIFIED",
  "FAILED",
  "QUALITY_REJECTED",
  "CANCELLED",
];
const active = (job?: Job) =>
  Boolean(
    job && !terminal.includes(job.state) && job.state !== "WAITING_FOR_USER",
  );
const idOf = (job: Job) => job.id || job.job_id || "";
const errorText = (error: unknown) =>
  error instanceof ApiError
    ? error.message
    : error instanceof Error
      ? error.message
      : "Something went wrong. Please try again.";
const fileKey = (name: string, hash: string) => name + "\u0000" + hash;
const bytes = (value: number) =>
  value > 1000000
    ? (value / 1000000).toFixed(1) + " MB"
    : Math.ceil(value / 1000) + " KB";
const wait = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function Shell({
  children,
  logout,
}: {
  children: React.ReactNode;
  logout: () => void;
}) {
  return (
    <div className={s.shell}>
      <aside className={s.sidebar}>
        <div>
          <div className={s.brand}>
            <span className={s.mark}>L</span> LoRA Training Studio
          </div>
          <p className={s.tagline}>Guided model training</p>
        </div>
        <nav className={s.nav}>
          <NavLink to="/runs">Training Runs</NavLink>
          <NavLink to="/new">New Training</NavLink>
          <NavLink to="/system">System Status</NavLink>
        </nav>
        <div className={s.sidebarFooter}>
          <span>Local, private workspace</span>
          <button className={s.logout} onClick={logout}>
            Sign out
          </button>
        </div>
      </aside>
      <main className={s.main}>{children}</main>
    </div>
  );
}
function Auth({ success }: { success: () => void }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!value.trim())
      return setError("Enter the API token configured for this server.");
    sessionStorage.setItem("lora-token", value.trim());
    try {
      await apiClient.profiles();
      success();
      navigate("/runs");
    } catch (reason) {
      sessionStorage.removeItem("lora-token");
      setError(errorText(reason));
    }
  };
  return (
    <div className={s.auth}>
      <form className={s.authCard} onSubmit={submit}>
        <div className={s.brand} style={{ color: "#102a43" }}>
          <span className={s.mark}>L</span> LoRA Training Studio
        </div>
        <h1>Connect your workspace</h1>
        <p className={s.sub}>
          Use the bearer token from your local pipeline configuration. It stays
          in this browser session only.
        </p>
        <div className={s.form}>
          <label className={s.field}>
            API token
            <input
              aria-label="API token"
              autoFocus
              type="password"
              value={value}
              onChange={(event) => setValue(event.target.value)}
            />
          </label>
          {error && <div className={s.error}>{error}</div>}
          <button className={s.button}>Connect</button>
        </div>
      </form>
    </div>
  );
}
function Badge({ state }: { state: string }) {
  return (
    <span className={s.state} data-state={state}>
      {state.replaceAll("_", " ").toLowerCase()}
    </span>
  );
}

function Runs() {
  const navigate = useNavigate();
  const [state, setState] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 25;
  const query = useQuery({
    queryKey: ["jobs", state, offset],
    queryFn: () => apiClient.jobs(state || undefined, limit, offset),
    refetchInterval: (q) => (q.state.data?.items.some(active) ? 2000 : false),
  });
  const jobs = query.data?.items || [];
  return (
    <>
      <header className={s.top}>
        <div>
          <p className={s.eyebrow}>Workspace</p>
          <h1>Training runs</h1>
          <p className={s.sub}>
            Create, monitor, and publish your private LoRA adapters.
          </p>
        </div>
        <button className={s.button} onClick={() => navigate("/new")}>
          New training
        </button>
      </header>
      <div className={s.cards}>
        <Stat
          title="Active runs"
          value={String(jobs.filter(active).length)}
          detail="Training or processing now"
        />
        <Stat
          title="Awaiting you"
          value={String(
            jobs.filter((job) => job.state === "WAITING_FOR_USER").length,
          )}
          detail="Ready for the next stage"
        />
        <Stat
          title="All visible runs"
          value={String(query.data?.total ?? "—")}
          detail="Your training history"
        />
      </div>
      <section className={s.tableCard + " " + s.card}>
        <div className={s.tableTitle}>
          <h2>Recent runs</h2>
          <label className={s.field}>
            Status
            <select
              value={state}
              onChange={(event) => {
                setState(event.target.value);
                setOffset(0);
              }}
            >
              <option value="">All statuses</option>
              <option value="WAITING_FOR_USER">Awaiting action</option>
              <option value="ACCEPTED">Queued</option>
              <option value="RUNNING">Running</option>
              <option value="READY">Published</option>
              <option value="COMPLETED_UNVERIFIED">Unverified</option>
              <option value="QUALITY_REJECTED">Quality rejected</option>
              <option value="FAILED">Failed</option>
              <option value="CANCELLED">Cancelled</option>
            </select>
          </label>
        </div>
        {query.isError ? (
          <div className={s.error}>{errorText(query.error)}</div>
        ) : query.isLoading ? (
          <div className={s.loading}>Loading runs…</div>
        ) : !jobs.length ? (
          <div className={s.empty}>No training runs match this filter.</div>
        ) : (
          <>
            <table className={s.table}>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Dataset</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    data-clickable="true"
                    key={idOf(job)}
                    onClick={() => navigate("/runs/" + idOf(job))}
                  >
                    <td>
                      <strong>{idOf(job).slice(0, 8)}</strong>
                      <br />
                      <span className={s.muted}>
                        {job.trigger_token || "LoRA adapter"}
                      </span>
                    </td>
                    <td>
                      <Badge state={job.state} />
                    </td>
                    <td>{job.dataset_id.slice(0, 8)}</td>
                    <td>
                      {job.updated_at
                        ? new Date(
                            typeof job.updated_at === "number"
                              ? job.updated_at * 1000
                              : job.updated_at,
                          ).toLocaleString()
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className={s.pager}>
              <button
                className={s.buttonSecondary + " " + s.button}
                disabled={!offset}
                onClick={() => setOffset(Math.max(0, offset - limit))}
              >
                Previous
              </button>
              <span className={s.muted}>
                {offset + 1}–{offset + jobs.length} of {query.data?.total}
              </span>
              <button
                className={s.buttonSecondary + " " + s.button}
                disabled={offset + limit >= (query.data?.total || 0)}
                onClick={() => setOffset(offset + limit)}
              >
                Next
              </button>
            </div>
          </>
        )}
      </section>
    </>
  );
}
function Stat({
  title,
  value,
  detail,
}: {
  title: string;
  value: string;
  detail: string;
}) {
  return (
    <div className={s.card}>
      <h3>{title}</h3>
      <div className={s.stat}>{value}</div>
      <span className={s.muted}>{detail}</span>
    </div>
  );
}

type Local = {
  file: File;
  hash?: string;
  caption: string;
  progress: number;
  preview: string;
};
type Pending = {
  id: string;
  profileId: string;
  trigger: string;
  intent: string;
  files: Array<{
    name: string;
    hash: string;
    size: number;
    type: string;
    caption: string;
  }>;
};
const pendingStorage = "lora-pending-upload";
const readPending = () => {
  try {
    const item = sessionStorage.getItem(pendingStorage);
    return item ? (JSON.parse(item) as Pending) : undefined;
  } catch {
    return undefined;
  }
};
const savePending = (value?: Pending) =>
  value
    ? sessionStorage.setItem(pendingStorage, JSON.stringify(value))
    : sessionStorage.removeItem(pendingStorage);
async function hashes(files: File[], signal: AbortSignal) {
  if (!window.isSecureContext || !crypto.subtle)
    throw new Error("Image hashing requires HTTPS or localhost.");
  const worker = new Worker(new URL("./hash.worker.ts", import.meta.url), {
    type: "module",
  });
  try {
    const result: string[] = [];
    for (const file of files) {
      if (signal.aborted)
        throw new DOMException("Hashing cancelled", "AbortError");
      result.push(
        await new Promise<string>((resolve, reject) => {
          const request = crypto.randomUUID();
          const abort = () =>
            reject(new DOMException("Hashing cancelled", "AbortError"));
          signal.addEventListener("abort", abort, { once: true });
          worker.onmessage = (
            event: MessageEvent<{
              id: string;
              sha256?: string;
              error?: string;
            }>,
          ) => {
            if (event.data.id === request)
              event.data.error
                ? reject(new Error(event.data.error))
                : resolve(event.data.sha256 || "");
          };
          worker.postMessage({ id: request, file });
        }),
      );
    }
    return result;
  } finally {
    worker.terminate();
  }
}
async function captionMap(file: File) {
  const parsed: unknown = JSON.parse(await file.text());
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object")
    throw new Error("captions.json must map image file names to captions.");
  return Object.fromEntries(
    Object.entries(parsed as Record<string, unknown>).filter(
      ([, value]) => typeof value === "string",
    ),
  ) as Record<string, string>;
}

function NewTraining() {
  const navigate = useNavigate();
  const client = useQueryClient();
  const saved = useMemo(readPending, []);
  const [pending, setPending] = useState<Pending | undefined>(saved);
  const [profileId, setProfileId] = useState(saved?.profileId || "");
  const [trigger, setTrigger] = useState(saved?.trigger || "");
  const [files, setFiles] = useState<Local[]>([]);
  const filesRef = useRef<Local[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | undefined>(undefined);
  const automaticallyResumed = useRef("");
  // Incrementing this makes late completion handlers from an abandoned upload
  // harmless.  Aborting an XHR alone is not enough because dataset polling can
  // finish after the user has chosen to start over.
  const flowVersion = useRef(0);
  const intent = useRef(
    saved?.intent ||
      sessionStorage.getItem("lora-draft-intent") ||
      crypto.randomUUID(),
  );
  useEffect(() => {
    sessionStorage.setItem("lora-draft-intent", intent.current);
  }, []);
  const profiles = useQuery({
    queryKey: ["profiles"],
    queryFn: apiClient.profiles,
  });
  const datasets = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.datasets(),
  });
  const pendingDataset = useQuery({
    queryKey: ["dataset", pending?.id],
    queryFn: () => apiClient.dataset(pending!.id),
    enabled: Boolean(pending),
    refetchInterval: (query) =>
      query.state.data?.state === "COMPLETED" ||
      query.state.data?.state === "INVALID"
        ? false
        : 1500,
  });
  const profile =
    profiles.data?.profiles.find(
      (item) => item.profile_revision_id === profileId,
    ) || profiles.data?.profiles[0];
  const data = profile?.data || {};
  const min = Number(data.min_images || 1);
  const max = Number(data.max_images || 1000);
  const maxFile = Number(data.max_file_bytes || Number.MAX_SAFE_INTEGER);
  const maxTotal = Number(data.max_total_bytes || Number.MAX_SAFE_INTEGER);
  useEffect(() => {
    if (profile && !profileId) setProfileId(profile.profile_revision_id);
  }, [profile, profileId]);
  useEffect(
    () => () => {
      controller.current?.abort();
    },
    [],
  );
  useEffect(() => {
    filesRef.current = files;
  }, [files]);
  useEffect(
    () => () => {
      filesRef.current.forEach((item) => URL.revokeObjectURL(item.preview));
    },
    [],
  );
  const ensureCurrentFlow = (
    version: number,
    signal?: AbortSignal,
  ) => {
    if (flowVersion.current !== version || signal?.aborted)
      throw new DOMException("Upload cancelled", "AbortError");
  };
  const startFreshDraft = (message = "") => {
    const previousId = pending?.id;
    flowVersion.current += 1;
    controller.current?.abort();
    controller.current = undefined;
    automaticallyResumed.current = "";
    filesRef.current.forEach((item) => URL.revokeObjectURL(item.preview));
    filesRef.current = [];
    savePending();
    sessionStorage.removeItem("lora-draft-intent");
    intent.current = crypto.randomUUID();
    sessionStorage.setItem("lora-draft-intent", intent.current);
    if (previousId) {
      void client.cancelQueries({
        queryKey: ["dataset", previousId],
        exact: true,
      });
      client.removeQueries({ queryKey: ["dataset", previousId], exact: true });
    }
    setPending(undefined);
    setFiles([]);
    setProfileId(profiles.data?.profiles[0]?.profile_revision_id || "");
    setTrigger("");
    setError(message);
    setBusy(false);
  };
  useEffect(() => {
    if (
      pending &&
      pendingDataset.data?.id === pending.id &&
      pendingDataset.data.state === "INVALID"
    )
      startFreshDraft(
        "Dataset validation rejected this upload. The upload was discarded; choose corrected images to start again.",
      );
  }, [pending, pendingDataset.data]);
  const validate = (candidate: File[]) => {
    if (candidate.length < min)
      return "This profile requires at least " + min + " images.";
    if (candidate.length > max)
      return "This profile accepts at most " + max + " images.";
    const oversized = candidate.find((file) => file.size > maxFile);
    if (oversized)
      return (
        oversized.name + " exceeds the " + bytes(maxFile) + " per-file limit."
      );
    if (candidate.reduce((total, file) => total + file.size, 0) > maxTotal)
      return "Images exceed the " + bytes(maxTotal) + " dataset limit.";
    return "";
  };
  const add = async (picked: FileList | null) => {
    if (!picked || busy || pending) return;
    const selected = [...picked];
    const captions = selected.filter(
      (file) => file.name.toLowerCase() === "captions.json",
    );
    const images = selected.filter((file) =>
      /^(image\/jpeg|image\/png|image\/webp)$/.test(file.type),
    );
    setError("");
    if (
      captions.length > 1 ||
      captions.length + images.length !== selected.length
    )
      return setError("Choose images and at most one captions.json file.");
    setBusy(true);
    try {
      const map = captions[0] ? await captionMap(captions[0]) : {};
      const names = new Set(
        files
          .map((item) => item.file.name)
          .concat(images.map((file) => file.name)),
      );
      if (names.size !== files.length + images.length)
        return setError("File names must be unique.");
      if (captions[0] && files.length)
        setFiles((old) =>
          old.map((item) =>
            Object.prototype.hasOwnProperty.call(map, item.file.name)
              ? { ...item, caption: map[item.file.name] }
              : item,
          ),
        );
      if (!images.length) return;
      const candidate = files.map((item) => item.file).concat(images);
      const oversized = candidate.find((file) => file.size > maxFile);
      const problem =
        candidate.length > max
          ? "This profile accepts at most " + max + " images."
          : oversized
            ? oversized.name +
              " exceeds the " +
              bytes(maxFile) +
              " per-file limit."
            : candidate.reduce((total, file) => total + file.size, 0) > maxTotal
              ? "Images exceed the " + bytes(maxTotal) + " dataset limit."
              : "";
      if (problem) return setError(problem);
      const current = new AbortController();
      controller.current = current;
      const digest = await hashes(images, current.signal);
      setFiles((old) =>
        old.concat(
          images.map((file, index) => ({
            file,
            hash: digest[index],
            caption: map[file.name] || "",
            progress: 0,
            preview: URL.createObjectURL(file),
          })),
        ),
      );
    } catch (reason) {
      if ((reason as Error).name !== "AbortError") setError(errorText(reason));
    } finally {
      setBusy(false);
    }
  };
  const createJob = async (
    datasetId: string,
    useProfile = profileId,
    useTrigger = trigger,
    requestIntent = intent.current,
    version = flowVersion.current,
    signal?: AbortSignal,
  ) => {
    ensureCurrentFlow(version, signal);
    if (!useTrigger.trim())
      throw new Error("Add a trigger token before creating a training run.");
    const job = await apiClient.createJob(
      {
        dataset_id: datasetId,
        profile_revision_id: useProfile,
        trigger_token: useTrigger.trim(),
        execution_mode: "manual",
      },
      "job:" + requestIntent,
    );
    ensureCurrentFlow(version, signal);
    savePending();
    sessionStorage.removeItem("lora-draft-intent");
    setPending(undefined);
    navigate("/runs/" + idOf(job));
  };
  const waitCompleted = async (
    datasetId: string,
    version: number,
    signal: AbortSignal,
  ) => {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      ensureCurrentFlow(version, signal);
      const item = await apiClient.dataset(datasetId);
      ensureCurrentFlow(version, signal);
      if (item.state === "COMPLETED") return;
      if (item.state === "INVALID")
        throw new Error(
          String(item.verification_error || "Dataset validation failed."),
        );
      await wait(1000);
    }
    throw new Error(
      "Dataset verification is still running. Return to this page to resume.",
    );
  };
  const resume = async (picked?: FileList | null) => {
    if (!pending) return;
    const resumePending = pending;
    const version = flowVersion.current;
    setBusy(true);
    setError("");
    const current = new AbortController();
    controller.current = current;
    try {
      let local = files;
      if (picked) {
        filesRef.current.forEach((item) => URL.revokeObjectURL(item.preview));
        const raw = [...picked].filter((file) =>
          /^(image\/jpeg|image\/png|image\/webp)$/.test(file.type),
        );
        const digest = await hashes(raw, current.signal);
        ensureCurrentFlow(version, current.signal);
        local = raw.map((file, index) => ({
          file,
          hash: digest[index],
          caption: "",
          progress: 0,
          preview: URL.createObjectURL(file),
        }));
        setFiles(local);
      }
      ensureCurrentFlow(version, current.signal);
      const remote = await apiClient.dataset(resumePending.id);
      ensureCurrentFlow(version, current.signal);
      if (remote.state === "COMPLETED")
        return await createJob(
          resumePending.id,
          resumePending.profileId,
          resumePending.trigger,
          resumePending.intent,
          version,
          current.signal,
        );
      if (remote.state === "INVALID")
        throw new Error(
          String(remote.verification_error || "Dataset validation failed."),
        );
      const source = new Map(
        local
          .filter((item) => item.hash)
          .map((item) => [fileKey(item.file.name, item.hash!), item]),
      );
      const missing = (remote.files || []).filter(
        (item) =>
          !item.uploaded && !source.has(fileKey(item.name, item.sha256)),
      );
      if (missing.length)
        throw new Error(
          "Choose the " +
            missing.length +
            " missing local file(s) to resume. Files are matched by name and SHA-256.",
        );
      const needed = (remote.files || []).filter((item) => !item.uploaded);
      let cursor = 0;
      const worker = async () => {
        while (cursor < needed.length) {
          const entry = needed[cursor++];
          const localFile = source.get(fileKey(entry.name, entry.sha256))!;
          const key = fileKey(localFile.file.name, localFile.hash!);
          await upload(
            resumePending.id,
            entry.id,
            localFile.file,
            (progress) => {
              if (flowVersion.current !== version || current.signal.aborted)
                return;
              setFiles((old) =>
                old.map((item) =>
                  item.hash && fileKey(item.file.name, item.hash) === key
                    ? { ...item, progress }
                    : item,
                ),
              );
            },
            current.signal,
          );
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(4, needed.length) }, worker),
      );
      ensureCurrentFlow(version, current.signal);
      await apiClient.completeDataset(resumePending.id);
      await waitCompleted(resumePending.id, version, current.signal);
      await createJob(
        resumePending.id,
        resumePending.profileId,
        resumePending.trigger,
        resumePending.intent,
        version,
        current.signal,
      );
    } catch (reason) {
      if (
        flowVersion.current === version &&
        (reason as Error).name !== "AbortError"
      )
        setError(errorText(reason));
    } finally {
      if (flowVersion.current === version) setBusy(false);
    }
  };
  const process = async () => {
    if (!trigger.trim())
      return setError("Add a trigger token before processing the dataset.");
    const problem = validate(files.map((item) => item.file));
    if (problem) return setError(problem);
    if (files.some((item) => !item.hash))
      return setError("Wait for image hashing to finish.");
    setBusy(true);
    setError("");
    try {
      const dataset = await apiClient.createDataset(
        files.map((item) => ({
          name: item.file.name,
          size_bytes: item.file.size,
          sha256: item.hash,
          mime_type: item.file.type,
          caption: item.caption || undefined,
        })),
        "dataset:" + intent.current,
      );
      const next: Pending = {
        id: String(dataset.id),
        profileId,
        trigger: trigger.trim(),
        intent: intent.current,
        files: files.map((item) => ({
          name: item.file.name,
          hash: item.hash!,
          size: item.file.size,
          type: item.file.type,
          caption: item.caption,
        })),
      };
      savePending(next);
      setPending(next);
      setBusy(false);
    } catch (reason) {
      setError(errorText(reason));
      setBusy(false);
    }
  };
  useEffect(() => {
    if (
      pending &&
      !busy &&
      files.length &&
      pending.id &&
      automaticallyResumed.current !== pending.id
    ) {
      automaticallyResumed.current = pending.id;
      void resume();
    }
  }, [pending?.id, busy, files.length]);
  const verified =
    datasets.data?.items.filter((item) => item.state === "COMPLETED") || [];
  return (
    <>
      <header className={s.top}>
        <div>
          <p className={s.eyebrow}>Step 1 of 4</p>
          <h1>Prepare a dataset</h1>
          <p className={s.sub}>
            Review captions before processing. Training starts only after your
            confirmation.
          </p>
        </div>
      </header>
      <div className={s.workspace + " " + s.section}>
        <div className={s.split}>
          <label className={s.field}>
            Training profile
            <select
              value={profileId}
              disabled={busy || Boolean(pending)}
              onChange={(event) => setProfileId(event.target.value)}
            >
              {profiles.data?.profiles.map((item) => (
                <option
                  value={item.profile_revision_id}
                  key={item.profile_revision_id}
                >
                  {item.display_name ||
                    item.profile_key ||
                    item.profile_revision_id}
                </option>
              ))}
            </select>
            {profile?.config.backend === "tiny" && (
              <span className={s.notice}>
                Demo / Test Only — this CPU profile validates the workflow.
              </span>
            )}
          </label>
          <label className={s.field}>
            Trigger token
            <input
              aria-label="Trigger token"
              disabled={busy || Boolean(pending)}
              value={trigger}
              onChange={(event) => setTrigger(event.target.value)}
            />
          </label>
        </div>
        {pending && (
          <div
            className={s.notice + " " + s.recoveryNotice}
            data-testid="upload-recovery"
          >
            <div className={s.recoveryCopy}>
              <strong>Upload recovery available.</strong>
              <span>
                Dataset {pending.id.slice(0, 8)} is{" "}
                {pendingDataset.data?.state || "loading"}.
              </span>
              <span className={s.recoveryNote}>
                Discarding recovery stops this browser from resuming the upload;
                it does not delete server data.
              </span>
            </div>
            <div className={s.recoveryActions} data-testid="upload-recovery-actions">
              <label className={s.recoveryFile}>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  multiple
                  disabled={busy}
                  onChange={(event) => void resume(event.target.files)}
                />
                Choose files to resume
              </label>
              <button
                className={s.button + " " + s.recoveryResume}
                disabled={busy}
                onClick={() => void resume()}
              >
                Resume selected files
              </button>
              <button
                className={s.button + " " + s.recoveryDiscard}
                onClick={() => startFreshDraft()}
              >
                Discard upload and start new
              </button>
            </div>
          </div>
        )}
        <label
          className={s.uploadZone}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            void add(event.dataTransfer.files);
          }}
        >
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,.json"
            multiple
            disabled={busy || Boolean(pending)}
            onChange={(event) => void add(event.target.files)}
          />
          <strong>Drop images here or choose files</strong>
          <span className={s.muted}>
            JPG, PNG, or WebP · optional captions.json · {min}–{max} images
          </span>
        </label>
        {error && <div className={s.error}>{error}</div>}
        {files.length > 0 && (
          <div className={s.fileGrid}>
            {files.map((item, index) => (
              <div className={s.fileRow} key={item.file.name}>
                <img className={s.thumbnail} src={item.preview} alt="" />
                <div className={s.fileHeader}>
                  <span className={s.fileName}>{item.file.name}</span>
                  <button
                    className={s.remove}
                    disabled={busy || Boolean(pending)}
                    onClick={() =>
                      setFiles((old) => {
                        URL.revokeObjectURL(old[index].preview);
                        return old.filter((entry) => entry !== item);
                      })
                    }
                  >
                    ×
                  </button>
                </div>
                <span className={s.muted}>
                  {bytes(item.file.size)} · {item.hash ? "hashed" : "hashing"}
                </span>
                <textarea
                  aria-label={"Caption for " + item.file.name}
                  disabled={busy || Boolean(pending)}
                  value={item.caption}
                  placeholder="Leave blank to generate a caption"
                  onChange={(event) =>
                    setFiles((old) =>
                      old.map((entry) =>
                        entry === item
                          ? { ...entry, caption: event.target.value }
                          : entry,
                      ),
                    )
                  }
                />
                {item.progress > 0 && (
                  <div className={s.progress}>
                    <i
                      style={{
                        width:
                          Math.round((item.progress / item.file.size) * 100) +
                          "%",
                      }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        <div className={s.actions}>
          <span className={s.muted}>
            {files.length} images ·{" "}
            {bytes(files.reduce((total, item) => total + item.file.size, 0))}
          </span>
          <button
            className={s.button}
            disabled={busy || Boolean(pending) || !files.length}
            onClick={() => void process()}
          >
            {busy ? "Processing…" : "Process dataset"}
          </button>
        </div>
      </div>
      {verified.length > 0 && (
        <section className={s.section}>
          <h2>Reuse a verified dataset</h2>
          {verified.map((dataset) => (
            <div className={s.row} key={dataset.id}>
              <span className={s.grow}>
                <strong>{dataset.id.slice(0, 8)}</strong>{" "}
                <span className={s.muted}>
                  {dataset.valid_count || dataset.file_count || "—"} verified
                  images
                </span>
              </span>
              <button
                className={s.buttonSecondary + " " + s.button}
                disabled={busy}
                onClick={() => {
                  setBusy(true);
                  void createJob(dataset.id)
                    .catch((reason) => setError(errorText(reason)))
                    .finally(() => setBusy(false));
                }}
              >
                Use this dataset
              </button>
            </div>
          ))}
        </section>
      )}
    </>
  );
}

const stages: Array<{ id: Stage; label: string }> = (
  ["PREPARE", "TRAIN", "EVALUATE", "PUBLISH"] as Stage[]
).map((id) => ({ id, label: stageLabels[id] }));
function StageNav({
  job,
  selected,
  select,
}: {
  job: Job;
  selected: Stage;
  select: (stage: Stage) => void;
}) {
  const raw = String(job.waiting_for_stage || job.current_stage || "PREPARE");
  const current = stages.findIndex(
    (item) => item.id === (raw === "CAPTION" ? "PREPARE" : raw),
  );
  return (
    <div className={s.steps}>
      {stages.map((item, index) => {
        const model = stageModel(job, item.id);
        return (
        <button
          key={item.id}
          className={s.step}
          aria-current={selected === item.id ? "step" : undefined}
          disabled={index > current && !terminal.includes(job.state)}
          onClick={() => select(item.id)}
        >
          <span className={s.stepTitle}>
            <span className={s.stepNumber}>{index + 1}.</span>
            {item.label}
          </span>
          <span className={s.stepStatus} data-status={model.status}>
            {model.statusLabel}
          </span>
        </button>
        );
      })}
    </div>
  );
}
function StageProgress({ model }: { model: StageModel }) {
  const progress = model.progress;
  return (
    <section className={s.stageProgress} aria-live="polite">
      <div className={s.stageProgressTop}>
        <div>
          <p className={s.eyebrow}>Current view</p>
          <h2>{model.label}</h2>
          <p className={s.sub}>
            <strong>{model.statusLabel}.</strong> {model.description}
          </p>
        </div>
        <span className={s.stageStatus} data-status={model.status}>
          {model.statusLabel}
        </span>
      </div>
      <div className={s.stageMeta}>
        <span>
          <b>Phase</b> {model.phase}
        </span>
        <span>
          <b>Attempts</b> {model.attemptCount || "—"}
        </span>
      </div>
      {progress.kind === "determinate" ? (
        <>
          <div
            className={s.stageBar}
            role="progressbar"
            aria-label={`${model.label} progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress.percent)}
            aria-valuetext={`${Math.round(progress.percent)}%, ${progress.label}`}
          >
            <i style={{ width: `${progress.percent}%` }} />
          </div>
          <p className={s.muted}>
            {Math.round(progress.percent)}% · {progress.label}
          </p>
        </>
      ) : progress.kind === "indeterminate" ? (
        <>
          <div
            className={s.stageBar + " " + s.stageBarIndeterminate}
            role="progressbar"
            aria-label={`${model.label} progress`}
            aria-valuetext={progress.label}
          >
            <i />
          </div>
          <p className={s.muted}>{progress.label}</p>
        </>
      ) : (
        <p className={s.muted}>{progress.label}</p>
      )}
    </section>
  );
}
function ErrorCard({
  error,
  stage,
  attemptCount,
}: {
  error: ErrorModel;
  stage?: Stage;
  attemptCount?: number;
}) {
  return (
    <section className={s.errorCard} role="alert">
      <h2>Action needed</h2>
      <p>{error.message}</p>
      <dl className={s.errorMeta}>
        {error.code && <><dt>Code</dt><dd>{error.code}</dd></>}
        {error.httpStatus !== undefined && <><dt>HTTP status</dt><dd>{error.httpStatus}</dd></>}
        {error.retryable !== undefined && <><dt>Retryable</dt><dd>{error.retryable ? "Yes" : "No"}</dd></>}
        {error.requestId && <><dt>Request ID</dt><dd>{error.requestId}</dd></>}
        {stage && <><dt>Stage</dt><dd>{stageLabels[stage]}</dd></>}
        {attemptCount !== undefined && <><dt>Attempt</dt><dd>{attemptCount || "—"}</dd></>}
      </dl>
      {error.details !== undefined && (
        <details className={s.errorDetails}>
          <summary>Technical details</summary>
          <pre>{errorDetails(error.details)}</pre>
        </details>
      )}
    </section>
  );
}
function Metrics({
  job,
  stage = job.current_stage || "PREPARE",
}: {
  job: Job;
  stage?: string;
}) {
  const progress = job.stages?.[stage]?.progress || {};
  const current = progress.current ?? progress.global_step;
  const total = progress.total ?? progress.total_steps;
  const pairs: Array<[string, unknown]> = [
    ["Stage", stage || job.state],
    [
      "Progress",
      current !== undefined && total !== undefined
        ? String(current) + " / " + String(total)
        : "—",
    ],
    ["Loss", progress.loss ?? "—"],
    ["Samples / sec", progress.samples_per_second ?? "—"],
    [
      "GPU memory",
      progress.gpu_memory_allocated
        ? bytes(Number(progress.gpu_memory_allocated))
        : "—",
    ],
  ];
  return (
    <div className={s.metrics}>
      {pairs.map(([label, value]) => (
        <div className={s.metric} key={label}>
          <span className={s.muted}>{label}</span>
          <b>{String(value)}</b>
        </div>
      ))}
    </div>
  );
}
function Detail() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["job", id],
    queryFn: () => apiClient.job(id),
    refetchInterval: (q) => (active(q.state.data) ? 2000 : false),
  });
  const job = query.data;
  const [selected, setSelected] = useState<Stage>("PREPARE");
  const selectedInitialStage = useRef("");
  const [error, setError] = useState<unknown>(null);
  const [train, setTrain] = useState<Record<string, string>>({});
  const [evalValues, setEvalValues] = useState({
    prompts: "",
    seeds: "",
    inference_steps: "",
    guidance_scale: "",
  });
  useEffect(() => {
    if (!job) return;
    const config = job.profile?.config || {};
    setTrain({
      max_steps: String(config.max_steps || ""),
      learning_rate: String(config.learning_rate || ""),
      rank: String(config.rank || ""),
      lora_alpha: String(config.lora_alpha || ""),
      batch_size: String(config.batch_size || ""),
      gradient_accumulation_steps: String(
        config.gradient_accumulation_steps || "",
      ),
      checkpoint_every: String(config.checkpoint_every || ""),
      seed: String(config.seed ?? ""),
    });
    const evaluation = job.profile?.evaluation || {};
    setEvalValues({
      prompts: Array.isArray(evaluation.prompts)
        ? evaluation.prompts.join("\n")
        : "",
      seeds: Array.isArray(evaluation.seeds) ? evaluation.seeds.join(",") : "",
      inference_steps: String(evaluation.inference_steps || ""),
      guidance_scale: String(evaluation.guidance_scale ?? ""),
    });
  }, [job?.id]);
  useEffect(() => {
    const raw = String(job?.waiting_for_stage || job?.current_stage || "");
    const stage = raw === "CAPTION" ? "PREPARE" : raw;
    if (
      stage &&
      stages.some((item) => item.id === stage) &&
      (!selectedInitialStage.current || job?.waiting_for_stage)
    ) {
      selectedInitialStage.current = stage;
      setSelected(stage as Stage);
    }
  }, [job?.id, job?.current_stage, job?.waiting_for_stage]);
  useEffect(() => {
    if (job?.waiting_for_stage) setSelected(job.waiting_for_stage);
  }, [job?.waiting_for_stage]);
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["job", id] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  };
  const advance = useMutation({
    mutationFn: (body: unknown) => apiClient.advance(id, body),
    onSuccess: () => {
      setError(null);
      refresh();
    },
    onError: setError,
  });
  const cancel = useMutation({
    mutationFn: () => apiClient.cancel(id),
    onSuccess: () => {
      setError(null);
      refresh();
    },
    onError: setError,
  });
  if (query.isLoading)
    return <div className={s.loading}>Loading training run…</div>;
  if (query.isError || !job)
    return (
      <div className={s.workspace}>
        <ErrorCard error={formatError(query.error)} />
        <button className={s.button} onClick={() => navigate("/runs")}>
          Back to runs
        </button>
      </div>
    );
  const waiting = job.waiting_for_stage;
  const selectedModel = stageModel(job, selected);
  const currentStage = job.current_stage === "CAPTION" ? "PREPARE" :
    job.current_stage && stages.some((item) => item.id === job.current_stage)
      ? job.current_stage
      : selected;
  const currentModel = stageModel(job, currentStage as Stage);
  const failure = jobError(job, currentStage as Stage);
  const validTrain = () => {
    const ranges: Record<string, [number, number, boolean]> = {
      max_steps: [1, 100000, true],
      learning_rate: [0.0000001, 0.1, false],
      rank: [1, 64, true],
      lora_alpha: [1, 128, true],
      batch_size: [1, 16, true],
      gradient_accumulation_steps: [1, 128, true],
      checkpoint_every: [1, 100000, true],
      seed: [0, Number.MAX_SAFE_INTEGER, true],
    };
    for (const [name, limit] of Object.entries(ranges)) {
      const value = Number(train[name]);
      if (
        !Number.isFinite(value) ||
        value < limit[0] ||
        value > limit[1] ||
        (limit[2] && !Number.isInteger(value))
      )
        return "Invalid " + name.replaceAll("_", " ") + ".";
    }
    return "";
  };
  const startTrain = () => {
    const problem = validTrain();
    if (problem) return setError(new Error(problem));
    setError(null);
    advance.mutate({
      stage: "TRAIN",
      training_overrides: Object.fromEntries(
        Object.entries(train).map(([name, value]) => [name, Number(value)]),
      ),
    });
  };
  const startEval = () => {
    const prompts = evalValues.prompts
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);
    const seeds = evalValues.seeds
      .split(",")
      .map((item) => Number(item.trim()));
    const inference = Number(evalValues.inference_steps);
    const guidance = Number(evalValues.guidance_scale);
    if (
      !evalValues.seeds.trim() ||
      evalValues.seeds.split(",").some((item) => !item.trim()) ||
      !prompts.length ||
      prompts.some((item) => item.length > 512) ||
      !seeds.length ||
      seeds.some((item) => !Number.isSafeInteger(item) || item < 0) ||
      prompts.length * seeds.length > 100 ||
      !Number.isInteger(inference) ||
      inference < 1 ||
      inference > 100 ||
      !Number.isFinite(guidance) ||
      guidance < 0 ||
      guidance > 30
    )
      return setError(new Error(
        "Enter valid prompts, non-negative integer seeds, 1–100 inference steps, and guidance from 0–30.",
      ));
    setError(null);
    advance.mutate({
      stage: "EVALUATE",
      evaluation_overrides: {
        prompts,
        seeds,
        inference_steps: inference,
        guidance_scale: guidance,
      },
    });
  };
  const modelId = job.model_id || job.model?.id;
  return (
    <>
      <header className={s.top}>
        <div>
          <p className={s.eyebrow}>Training run · {idOf(job).slice(0, 8)}</p>
          <h1>{job.trigger_token || "LoRA adapter"}</h1>
          <p className={s.sub}>
            Dataset {job.dataset_id.slice(0, 8)} · <Badge state={job.state} />
          </p>
        </div>
        {!terminal.includes(job.state) && (
          <button
            className={
              s.buttonDanger + " " + s.buttonSecondary + " " + s.button
            }
            disabled={cancel.isPending}
            onClick={() => {
              setError(null);
              cancel.mutate();
            }}
          >
            Cancel run
          </button>
        )}
      </header>
      {job.profile?.config.backend === "tiny" && (
        <p className={s.notice}>
          Demo / Test Only — this CPU run verifies the workflow, not SD 1.5
          image quality.
        </p>
      )}
      <StageNav job={job} selected={selected} select={setSelected} />
      <StageProgress model={selectedModel} />
      {error !== null && <ErrorCard error={formatError(error)} />}
      {failure && (
        <ErrorCard
          error={failure}
          stage={currentStage as Stage}
          attemptCount={currentModel.attemptCount}
        />
      )}
      <div className={s.workspace}>
        {selected === "PREPARE" && (
          <section className={s.section}>
            <h2>Prepared dataset</h2>
            <p className={s.sub}>
              Frozen images and final captions for this training run.
            </p>
            <PreparedStats job={job} />
            <Images job={job} kind="prepared_image" />
          </section>
        )}
        {selected === "TRAIN" && (
          <>
            {waiting === "TRAIN" && (
              <section className={s.section}>
                <h2>Confirm training setup</h2>
                <p className={s.sub}>
                  These values freeze when training begins.
                </p>
                <div className={s.paramGrid}>
                  {[
                    "max_steps",
                    "learning_rate",
                    "batch_size",
                    "gradient_accumulation_steps",
                  ].map((name) => (
                    <Field
                      key={name}
                      label={name}
                      value={train[name] || ""}
                      set={(value) => setTrain({ ...train, [name]: value })}
                    />
                  ))}
                </div>
                <details className={s.advanced}>
                  <summary>Advanced LoRA settings</summary>
                  <div className={s.paramGrid}>
                    {["rank", "lora_alpha", "checkpoint_every", "seed"].map(
                      (name) => (
                        <Field
                          key={name}
                          label={name}
                          value={train[name] || ""}
                          set={(value) => setTrain({ ...train, [name]: value })}
                        />
                      ),
                    )}
                  </div>
                </details>
                <div className={s.actions}>
                  <span className={s.muted}>
                    Compute capacity is reserved after you start.
                  </span>
                  <button
                    className={s.button}
                    disabled={advance.isPending}
                    onClick={startTrain}
                  >
                    Start training
                  </button>
                </div>
              </section>
            )}
            {(active(job) || job.current_stage === "TRAIN" || job.training) && (
              <section className={s.section}>
                <h2>Training progress</h2>
                <Metrics job={job} stage="TRAIN" />
                <ArtifactDownloads job={job} kinds={["training_manifest"]} />
              </section>
            )}
          </>
        )}
        {selected === "EVALUATE" && (
          <>
            {waiting === "EVALUATE" && (
              <section className={s.section}>
                <h2>Run evaluation</h2>
                <label className={s.field}>
                  Prompts
                  <textarea
                    value={evalValues.prompts}
                    onChange={(event) =>
                      setEvalValues({
                        ...evalValues,
                        prompts: event.target.value,
                      })
                    }
                  />
                </label>
                <div className={s.paramGrid}>
                  <Field
                    label="Seeds"
                    value={evalValues.seeds}
                    set={(value) =>
                      setEvalValues({ ...evalValues, seeds: value })
                    }
                  />
                  <Field
                    label="Inference steps"
                    value={evalValues.inference_steps}
                    set={(value) =>
                      setEvalValues({ ...evalValues, inference_steps: value })
                    }
                  />
                  <Field
                    label="Guidance scale"
                    value={evalValues.guidance_scale}
                    set={(value) =>
                      setEvalValues({ ...evalValues, guidance_scale: value })
                    }
                  />
                </div>
                <button
                  className={s.button}
                  disabled={advance.isPending}
                  onClick={startEval}
                >
                  Run evaluation
                </button>
              </section>
            )}
            {(waiting === "PUBLISH" ||
              job.evaluation ||
              terminal.includes(job.state)) && <Evaluation job={job} />}
          </>
        )}
        {selected === "PUBLISH" && (
          <section className={s.section}>
            <h2>Publish & download</h2>
            {waiting === "PUBLISH" && (
              <button
                className={s.button}
                disabled={advance.isPending}
                onClick={() => {
                  setError(null);
                  advance.mutate({ stage: "PUBLISH" });
                }}
              >
                Publish model
              </button>
            )}
            {["READY", "COMPLETED_UNVERIFIED"].includes(job.state) &&
              modelId && (
                <Download
                  modelId={modelId}
                  unverified={job.state === "COMPLETED_UNVERIFIED"}
                />
              )}
            {job.state === "QUALITY_REJECTED" && (
              <div className={s.error}>
                Quality review did not pass, so this adapter cannot be
                published.
              </div>
            )}
          </section>
        )}
      </div>
    </>
  );
}
function Field({
  label,
  value,
  set,
}: {
  label: string;
  value: string;
  set: (value: string) => void;
}) {
  return (
    <label className={s.field}>
      {label.replaceAll("_", " ")}
      <input
        type={label === "Seeds" ? "text" : "number"}
        value={value}
        onChange={(event) => set(event.target.value)}
      />
    </label>
  );
}
function Evaluation({ job }: { job: Job }) {
  const query = useQuery({
    queryKey: ["evaluation", idOf(job)],
    queryFn: () => apiClient.evaluation(idOf(job)),
  });
  const report = query.data || {};
  const metrics =
    report.metrics && typeof report.metrics === "object"
      ? Object.entries(report.metrics as Record<string, unknown>)
      : [];
  if (query.isError)
    return <ErrorCard error={formatError(query.error)} stage="EVALUATE" />;
  return (
    <section className={s.section}>
      <h2>Evaluation results</h2>
      <div className={s.metrics}>
        <Stat
          title="Technical check"
          value={
            report.technical_pass === true
              ? "Passed"
              : report.technical_pass === false
                ? "Failed"
                : "—"
          }
          detail="Model load smoke test"
        />
        <Stat
          title="Quality status"
          value={String(report.quality_status || "—")}
          detail="Policy result"
        />
        {metrics
          .filter(
            ([, value]) =>
              value === null ||
              typeof value === "number" ||
              typeof value === "string",
          )
          .map(([name, value]) => (
            <Stat
              title={name.replaceAll("_", " ")}
              value={value === null ? "Unavailable" : String(value)}
              detail="Recorded metric"
              key={name}
            />
          ))}
      </div>
      <Images job={job} kind="evaluation_image" />
    </section>
  );
}
function Download({
  modelId,
  unverified,
}: {
  modelId: string;
  unverified: boolean;
}) {
  const [ack, setAck] = useState(false);
  const [url, setUrl] = useState("");
  const [error, setError] = useState<unknown>(null);
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );
  const prepare = async () => {
    if (unverified && !ack)
      return setError(
        new Error("Confirm the unverified download acknowledgement first."),
      );
    try {
      setError(null);
      setUrl(
        URL.createObjectURL(
          await apiClient.blob(
            "/v1/models/" +
              modelId +
              "/download" +
              (unverified ? "?allow_unverified=true" : ""),
          ),
        ),
      );
    } catch (reason) {
      setError(reason);
    }
  };
  return (
    <div>
      {unverified && (
        <label className={s.check}>
          <input
            type="checkbox"
            checked={ack}
            onChange={(event) => setAck(event.target.checked)}
          />{" "}
          I understand this model has not completed verification.
        </label>
      )}
      {error !== null && <ErrorCard error={formatError(error)} />}
      <button className={s.button} onClick={() => void prepare()}>
        Prepare adapter download
      </button>
      {url && (
        <a
          className={s.download}
          href={url}
          download={modelId + ".safetensors"}
        >
          Download adapter
        </a>
      )}
    </div>
  );
}
function System() {
  const query = useQuery({
    queryKey: ["health"],
    queryFn: apiClient.ready,
    refetchInterval: 10000,
  });
  const status = query.data;
  return (
    <>
      <header className={s.top}>
        <div>
          <p className={s.eyebrow}>Operations</p>
          <h1>System status</h1>
          <p className={s.sub}>Live health information from this deployment.</p>
        </div>
      </header>
      {query.isError ? (
        <div className={s.error}>{errorText(query.error)}</div>
      ) : (
        <div className={s.cards}>
          <Stat
            title="API"
            value={status?.status || "checking"}
            detail="API readiness"
          />
          <Stat
            title="Database"
            value={status?.database ? "connected" : "unavailable"}
            detail="Persistent state"
          />
          <Stat
            title="Storage"
            value={status?.storage ? "writable" : "unavailable"}
            detail="Artifacts and datasets"
          />
          <Stat
            title="Worker"
            value={status?.worker ? "ready" : "not ready"}
            detail="Stage executor"
          />
        </div>
      )}
    </>
  );
}
export default function App() {
  const [authenticated, setAuthenticated] = useState(
    Boolean(sessionStorage.getItem("lora-token")),
  );
  const client = useQueryClient();
  const logout = () => {
    abortPendingRequests();
    sessionStorage.clear();
    client.clear();
    setAuthenticated(false);
  };
  if (!authenticated)
    return (
      <Routes>
        <Route
          path="*"
          element={<Auth success={() => setAuthenticated(true)} />}
        />
      </Routes>
    );
  return (
    <Shell logout={logout}>
      <Routes>
        <Route path="/runs" element={<Runs />} />
        <Route path="/runs/:id" element={<Detail />} />
        <Route path="/new" element={<NewTraining />} />
        <Route path="/system" element={<System />} />
        <Route path="*" element={<Navigate to="/runs" replace />} />
      </Routes>
    </Shell>
  );
}
