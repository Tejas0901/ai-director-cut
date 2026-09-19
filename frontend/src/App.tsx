import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteJob,
  downloadUrl,
  formatAge,
  formatTime,
  getHealth,
  getJob,
  listJobs,
  posterUrl,
  streamJob,
  uploadVideo,
  type Health,
  type Job,
  type JobSummary,
  type Stage,
} from "./api";

type Phase = "idle" | "working" | "done" | "error";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [stage, setStage] = useState("queued");
  const [label, setLabel] = useState("");
  const [detail, setDetail] = useState("");
  const [progress, setProgress] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [library, setLibrary] = useState<JobSummary[]>([]);
  const [pendingDelete, setPendingDelete] = useState<JobSummary | null>(null);

  const teardown = useRef<(() => void) | null>(null);
  const reelRef = useRef<HTMLVideoElement>(null);

  const refreshLibrary = useCallback(() => {
    listJobs()
      .then((all) => setLibrary(all.filter((j) => j.stage === "done")))
      .catch(() => setLibrary([]));
  }, []);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    refreshLibrary();
    return () => teardown.current?.();
  }, [refreshLibrary]);

  // Survive a page refresh mid-render: the job id lives in the URL hash, and
  // the SSE stream replays current state the moment we resubscribe.
  useEffect(() => {
    const existing = window.location.hash.replace("#", "").trim();
    if (existing) attach(existing);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function attach(jobId: string) {
    window.location.hash = jobId;
    setPhase("working");
    setError("");

    teardown.current?.();
    teardown.current = streamJob(
      jobId,
      (event) => {
        setStage(event.stage);
        setLabel(event.label);
        setProgress(event.progress);
        if (event.detail) setDetail(event.detail);
        if (event.stage === "failed") {
          setPhase("error");
          setError(event.detail || "the pipeline failed");
        }
      },
      (finished) => {
        setJob(finished);
        setPhase(finished.error ? "error" : "done");
        if (finished.error) setError(finished.error);
        else refreshLibrary(); // the reel we just cut belongs in the library
      },
      (message) => {
        // Fall back to a plain fetch: the job may well have finished while
        // the stream was dropping.
        getJob(jobId)
          .then((j) => {
            setJob(j);
            setPhase(j.stage === "done" ? "done" : "error");
            if (j.stage !== "done") setError(j.error ?? message);
          })
          .catch(() => {
            setPhase("error");
            setError(message);
          });
      },
    );
  }

  const handleFile = useCallback(async (file: File) => {
    setJob(null);
    setDetail("");
    setProgress(0);
    setStage("queued");
    setLabel("Uploading...");
    setPhase("working");
    try {
      const { job_id } = await uploadVideo(file);
      attach(job_id);
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : "upload failed");
    }
  }, []);

  // Deleting is irreversible and removes files from disk, so it goes through
  // a confirmation step. The reel awaiting an answer lives here rather than
  // inside the card, so only one dialog can ever be open.
  const confirmDelete = useCallback(async () => {
    const reel = pendingDelete;
    setPendingDelete(null);
    if (!reel) return;

    try {
      await deleteJob(reel.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "delete failed");
      setPhase("error");
      return;
    }
    refreshLibrary();
  }, [pendingDelete, refreshLibrary]);

  function reset() {
    teardown.current?.();
    window.location.hash = "";
    setJob(null);
    setPhase("idle");
    setError("");
    setProgress(0);
    setDetail("");
  }

  function seekTo(seconds: number) {
    const video = reelRef.current;
    if (!video) return;
    video.currentTime = seconds;
    void video.play();
  }

  const stages: Stage[] = health?.stages ?? [];

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>Final Cut AI</h1>
          <p className="tagline">Raw footage in. Highlight reel out. All on your laptop.</p>
        </div>
        {health && (
          <div className="providers" title="Active providers">
            <Pill k="LLM" v={health.llm} />
            <Pill k="STT" v={health.stt} />
            <Pill k="TTS" v={health.tts} />
          </div>
        )}
      </header>

      {!health && (
        <div className="banner warn">
          Backend unreachable. Start it with <code>uv run uvicorn backend.main:app --reload</code>
        </div>
      )}

      {health && health.stubbed_stages.length > 0 && (
        <div className="banner info">
          Skeleton mode - stubbed stages: {health.stubbed_stages.join(", ")}
        </div>
      )}

      {phase === "idle" && (
        <>
          <Dropzone dragging={dragging} setDragging={setDragging} onFile={handleFile} />
          <Library reels={library} onOpen={attach} onDelete={setPendingDelete} />
        </>
      )}

      {phase === "working" && (
        <section className="card progress-card">
          <div className="progress-head">
            <span className="spinner" aria-hidden />
            <div>
              <h2>{label || "Starting up..."}</h2>
              {detail && <p className="detail">{detail}</p>}
            </div>
            <span className="pct">{Math.round(progress * 100)}%</span>
          </div>

          <div className="bar">
            <div className="bar-fill" style={{ width: `${progress * 100}%` }} />
          </div>

          <ol className="stages">
            {stages.map((s) => {
              const currentIndex = stages.findIndex((x) => x.key === stage);
              const thisIndex = stages.findIndex((x) => x.key === s.key);
              const state =
                thisIndex < currentIndex ? "done" : thisIndex === currentIndex ? "active" : "todo";
              return (
                <li key={s.key} className={state}>
                  <span className="dot" />
                  {s.label}
                </li>
              );
            })}
          </ol>
        </section>
      )}

      {phase === "error" && (
        <section className="card error-card">
          <h2>That didn&rsquo;t work</h2>
          <pre>{error}</pre>
          <button onClick={reset}>Try another video</button>
        </section>
      )}

      {phase === "done" && job && (
        <Result job={job} reelRef={reelRef} onSeek={seekTo} onReset={reset} />
      )}

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete this reel?"
        confirmLabel="Delete reel"
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      >
        <strong>{pendingDelete?.title || pendingDelete?.filename}</strong> and its
        source upload will be removed from disk. This cannot be undone.
      </ConfirmDialog>
    </div>
  );
}

/**
 * Confirmation built on the native <dialog>, which brings focus trapping,
 * Esc-to-close and a real backdrop without a modal library. Cancel takes
 * focus rather than the destructive action, so a stray Enter does nothing.
 */
function ConfirmDialog({
  open,
  title,
  confirmLabel,
  onConfirm,
  onCancel,
  children,
}: {
  open: boolean;
  title: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // showModal() throws if called on an already-open dialog, so mirror the
    // prop onto the element rather than assuming they agree.
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="modal"
      onCancel={(e) => {
        e.preventDefault(); // let React own the open state, not the browser
        onCancel();
      }}
      // A click landing on the dialog itself is a click on the backdrop:
      // anything inside the panel targets the panel or its children.
      onClick={(e) => {
        if (e.target === ref.current) onCancel();
      }}
    >
      <div className="modal-panel">
        <h2>{title}</h2>
        <p>{children}</p>
        <div className="modal-actions">
          <button className="btn-quiet" onClick={onCancel} autoFocus>
            Cancel
          </button>
          <button className="btn-danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </dialog>
  );
}

function Pill({ k, v }: { k: string; v: string }) {
  return (
    <span className="pill">
      <b>{k}</b>
      {v}
    </span>
  );
}

function Dropzone({
  dragging,
  setDragging,
  onFile,
}: {
  dragging: boolean;
  setDragging: (v: boolean) => void;
  onFile: (f: File) => void;
}) {
  return (
    <label
      className={`dropzone${dragging ? " dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onFile(file);
      }}
    >
      <input
        type="file"
        accept="video/mp4,video/quicktime,video/x-matroska,video/webm"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
        }}
      />
      <div className="drop-icon">▶</div>
      <h2>Drop a video here</h2>
      <p>MP4, MOV, MKV or WebM. Keep it under two minutes for a fast first run.</p>
    </label>
  );
}

function Library({
  reels,
  onOpen,
  onDelete,
}: {
  reels: JobSummary[];
  onOpen: (id: string) => void;
  onDelete: (reel: JobSummary) => void;
}) {
  // Nothing cut yet is the normal first-run state, not an error worth a card.
  if (reels.length === 0) return null;

  return (
    <section className="card library">
      <div className="library-head">
        <h3>Your reels</h3>
        <p className="hint">
          {reels.length} saved on this machine. Click one to open it again.
        </p>
      </div>

      <ul className="reel-grid">
        {reels.map((reel) => (
          // The delete control is a sibling of the card, never nested inside
          // it - a button within a button is invalid and swallows the click.
          <li key={reel.id} className="reel-cell">
            <button
              className="reel-delete"
              title="Delete this reel"
              aria-label={`Delete ${reel.title || reel.filename}`}
              onClick={() => onDelete(reel)}
            >
              ×
            </button>
            <button className="reel" onClick={() => onOpen(reel.id)}>
              <span className="reel-thumb">
                <img
                  src={posterUrl(reel.id)}
                  alt=""
                  loading="lazy"
                  // A reel whose poster cannot be made still deserves a card,
                  // so fail to a blank tile rather than a broken-image icon.
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.visibility = "hidden";
                  }}
                />
                <span className="reel-play">▶</span>
              </span>
              <span className="reel-body">
                <span className="reel-title">{reel.title || "Untitled cut"}</span>
                <span className="reel-file">{reel.filename}</span>
                <span className="reel-meta">
                  {reel.clip_count} clip{reel.clip_count === 1 ? "" : "s"}
                  {reel.duration ? ` · from ${formatTime(reel.duration)}` : ""}
                  {formatAge(reel.created_at) ? ` · ${formatAge(reel.created_at)}` : ""}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Result({
  job,
  reelRef,
  onSeek,
  onReset,
}: {
  job: Job;
  reelRef: React.RefObject<HTMLVideoElement>;
  onSeek: (t: number) => void;
  onReset: () => void;
}) {
  const plan = job.plan;

  // Clip start times are positions in the SOURCE video. Inside the reel they
  // sit end to end, so the reel offset is the sum of all preceding durations.
  const reelOffsets: number[] = [];
  let running = 0;
  for (const clip of plan?.clips ?? []) {
    reelOffsets.push(running);
    running += clip.end_time - clip.start_time;
  }

  return (
    <>
      <section className="card result-head">
        <div>
          <p className="eyebrow">The cut</p>
          <h2>{plan?.title ?? "Your highlight reel"}</h2>
        </div>
        <div className="result-actions">
          {plan && <span className="pill"><b>mood</b>{plan.music_mood}</span>}
          {job.output_url && (
            <a className="btn-secondary" href={downloadUrl(job.id)}>
              Download reel
            </a>
          )}
          <button onClick={onReset}>New video</button>
        </div>
      </section>

      <section className="players">
        <figure>
          <figcaption>AI Highlight Reel</figcaption>
          <video ref={reelRef} src={job.output_url ?? undefined} controls playsInline />
        </figure>
        <figure>
          <figcaption>Original footage</figcaption>
          <video src={job.original_url ?? undefined} controls playsInline muted />
        </figure>
      </section>

      {plan && (
        <section className="grid">
          <div className="card">
            <h3>The Director&rsquo;s picks</h3>
            <p className="hint">Click a clip to jump to it in the reel.</p>
            <ul className="clips">
              {plan.clips.map((clip, i) => (
                <li key={i}>
                  <button className="clip" onClick={() => onSeek(reelOffsets[i])}>
                    <span className="clip-index">{i + 1}</span>
                    <span className="clip-body">
                      <span className="clip-title">{clip.overlay_title || "Untitled clip"}</span>
                      <span className="clip-reason">{clip.reason}</span>
                    </span>
                    <span className="clip-time">
                      {formatTime(clip.start_time)}&ndash;{formatTime(clip.end_time)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div className="card">
            <h3>Narration script</h3>
            <p className="script-label">Intro</p>
            <blockquote>{plan.intro_narration}</blockquote>
            <p className="script-label">Outro</p>
            <blockquote>{plan.outro_summary}</blockquote>

            {job.transcript && job.transcript.segments.length > 0 && (
              <>
                <h3 className="transcript-head">Source transcript</h3>
                <div className="transcript">
                  {job.transcript.segments.map((seg, i) => (
                    <p key={i}>
                      <span className="ts">{formatTime(seg.start)}</span>
                      {seg.text}
                    </p>
                  ))}
                </div>
              </>
            )}
          </div>
        </section>
      )}
    </>
  );
}
