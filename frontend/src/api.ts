// Types mirror backend/schemas.py. Keep them in sync by hand - the surface is
// small, and a codegen step is not worth the build complexity here.

export type Clip = {
  start_time: number;
  end_time: number;
  overlay_title: string;
  reason: string;
};

export type EditPlan = {
  title: string;
  music_mood: "energetic" | "chill" | "dramatic";
  intro_narration: string;
  outro_summary: string;
  clips: Clip[];
};

export type TranscriptSegment = { start: number; end: number; text: string };

export type Job = {
  id: string;
  filename: string;
  stage: string;
  label: string;
  progress: number;
  error: string | null;
  original_url: string | null;
  output_url: string | null;
  plan: EditPlan | null;
  transcript: { segments: TranscriptSegment[]; language: string } | null;
  duration: number | null;
};

export type JobEvent = {
  stage: string;
  label: string;
  progress: number;
  detail: string;
};

export type Stage = { key: string; label: string };

/** One card in the library. Deliberately lighter than a full Job. */
export type JobSummary = {
  id: string;
  filename: string;
  stage: string;
  title: string | null;
  output_url: string | null;
  clip_count: number;
  duration: number | null;
  created_at: number | null;
};

export type Health = {
  status: string;
  llm: string;
  stt: string;
  tts: string;
  stubbed_stages: string[];
  stages: Stage[];
};

export async function getHealth(): Promise<Health> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("backend unreachable");
  return res.json();
}

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/jobs", { method: "POST", body });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "upload failed");
  }
  return res.json();
}

export async function getJob(id: string): Promise<Job> {
  const res = await fetch(`/api/jobs/${id}`);
  if (!res.ok) throw new Error("job not found");
  return res.json();
}

/** Every reel cut on this machine, newest first. */
export async function listJobs(): Promise<JobSummary[]> {
  const res = await fetch("/api/jobs");
  if (!res.ok) throw new Error("could not load the library");
  return res.json();
}

export function posterUrl(id: string): string {
  return `/api/jobs/${id}/poster`;
}

/** The server sets Content-Disposition, so a plain link saves the file. */
export function downloadUrl(id: string): string {
  return `/api/jobs/${id}/download`;
}

/** Removes the reel, its source upload and its record. Not reversible. */
export async function deleteJob(id: string): Promise<void> {
  const res = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("could not delete that reel");
}

/**
 * Subscribe to a job's live progress.
 * Returns a teardown function; call it on unmount so a refresh mid-render
 * does not leave an orphaned EventSource retrying forever.
 */
export function streamJob(
  id: string,
  onEvent: (event: JobEvent) => void,
  onComplete: (job: Job) => void,
  onError: (message: string) => void,
): () => void {
  const source = new EventSource(`/api/jobs/${id}/events`);

  source.onmessage = (e) => onEvent(JSON.parse(e.data) as JobEvent);

  source.addEventListener("complete", (e) => {
    onComplete(JSON.parse((e as MessageEvent).data) as Job);
    source.close();
  });

  source.onerror = () => {
    // EventSource auto-reconnects on transient drops; only surface an error
    // once the connection is genuinely closed.
    if (source.readyState === EventSource.CLOSED) {
      onError("lost connection to the pipeline");
    }
  };

  return () => source.close();
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Relative age for a library card. Returns "" when the timestamp is missing —
 * reels cut before created_at existed have no honest answer, and inventing
 * one would be worse than showing nothing.
 */
export function formatAge(createdAt: number | null): string {
  if (!createdAt) return "";
  const seconds = Date.now() / 1000 - createdAt;
  if (seconds < 90) return "just now";
  const units: [number, string][] = [
    [60, "minute"],
    [3600, "hour"],
    [86400, "day"],
  ];
  for (let i = units.length - 1; i >= 0; i--) {
    const [size, name] = units[i];
    const n = Math.floor(seconds / size);
    if (n >= 1) return `${n} ${name}${n > 1 ? "s" : ""} ago`;
  }
  return "just now";
}
