export type FileRecord = {
  id: string;
  name: string;
  size_bytes: number;
  sha256: string;
  mime_type: string;
  caption?: string | null;
  uploaded?: boolean;
  verification_status?: string;
  rejection_code?: string;
};
export type Dataset = {
  id: string;
  state: string;
  files?: FileRecord[];
  file_count?: number;
  valid_count?: number;
  invalid_count?: number;
  verification_error?: string | null;
  created_at?: string;
  [key: string]: unknown;
};
export type Profile = {
  profile_revision_id: string;
  profile_key?: string;
  display_name?: string;
  config: Record<string, unknown>;
  data?: Record<string, unknown>;
  evaluation?: Record<string, unknown>;
  caption?: Record<string, unknown>;
};
export type Stage = "PREPARE" | "TRAIN" | "EVALUATE" | "PUBLISH";
export type StageTask = {
  state?: string;
  attempt_count?: number;
  progress?: Record<string, unknown>;
};
export type JobError = {
  message?: string;
  code?: string;
  details?: unknown;
  retryable?: boolean;
  request_id?: string;
};
export type Job = {
  id: string;
  job_id?: string;
  state: string;
  dataset_id: string;
  trigger_token?: string;
  profile_revision_id?: string;
  profile?: Profile;
  execution_mode?: string;
  waiting_for_stage?: Exclude<Stage, "PREPARE">;
  current_stage?: Stage | "CAPTION";
  stages?: Record<
    string,
    StageTask
  >;
  prepared?: Record<string, unknown> | null;
  input?: Record<string, unknown> | null;
  training?: Record<string, unknown> | null;
  evaluation?: Record<string, unknown> | null;
  error?: JobError | null;
  error_code?: string;
  error_message?: string;
  error_details?: Record<string, unknown> | null;
  model_id?: string;
  model?: { id?: string; state?: string };
  created_at?: string | number;
  updated_at?: string | number;
  [key: string]: unknown;
};
export type Artifact = {
  id: string;
  kind: string;
  label: string;
  media_type: string;
  url: string;
  name?: string;
  caption?: string;
  split?: string;
  prompt?: string;
  seed?: number;
  pair_index?: number;
  variant?: "base" | "adapter";
};
export type ApiList<T> = {
  items: T[];
  total: number;
  limit: number;
  offset: number;
};
export type Health = {
  status: string;
  database: boolean;
  storage: boolean;
  worker: boolean;
};
