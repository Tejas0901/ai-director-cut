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
