// Upload one file for background indexing and wait for its result.
//
// POST /upload with processing=background answers 202 and a job id; the job is
// then polled at GET /upload/jobs/{id} with the caller's bearer token until it
// completes or fails. The finished job carries the same batch envelope the
// synchronous route returns, so callers keep the existing "indexed only"
// acceptance rule. Jobs live in backend memory only: a 404 while polling means
// the job was lost (for example a server restart) and is reported as such,
// never as success.

export const UPLOAD_POLL_FIRST_MS = 1500;
export const UPLOAD_POLL_MAX_MS = 5000;
export const UPLOAD_MAX_WAIT_MS = 30 * 60 * 1000;
const MAX_POLL_NETWORK_ERRORS = 5;

export const UPLOAD_JOB_LOST =
  "The server lost track of this upload (it may have restarted). Check your documents list before uploading it again.";
export const UPLOAD_BUSY = "The server is busy with other uploads. Please try again in a minute.";
export const UPLOAD_WAIT_EXCEEDED =
  "Processing is taking longer than expected. Check your documents list later before uploading it again.";
export const UPLOAD_UNREACHABLE = "Could not reach the server to check on this upload. Please try again.";

const defaultSleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function describeJob(job) {
  if (job.status === "queued") {
    return job.queue_position > 1 ? `waiting (position ${job.queue_position})` : "waiting to start";
  }
  return "reading pages";
}

/**
 * Resolves to { result } (the first entry of the batch envelope, possibly not
 * indexed) or { error } (a message safe to show). Never rejects.
 */
export async function uploadAndWait({
  backend,
  file,
  authToken,
  extraction = "standard",
  onStatus = () => {},
  fetchImpl = (...args) => fetch(...args),
  sleep = defaultSleep,
  now = () => Date.now(),
  maxWaitMs = UPLOAD_MAX_WAIT_MS,
}) {
  const fd = new FormData();
  fd.append("files", file, file.name);
  if (authToken) fd.append("auth_token", authToken);
  fd.append("processing", "background");
  fd.append("extraction", extraction);

  let res;
  let data;
  try {
    onStatus("uploading");
    res = await fetchImpl(`${backend}/upload`, { method: "POST", body: fd });
    data = await res.json().catch(() => null);
  } catch (_) {
    return { error: null };
  }
  // A synchronous envelope (200/207/422/503) is still accepted as-is.
  if (data && Array.isArray(data.results)) return { result: data.results[0] || null };
  if (res.status === 429) return { error: UPLOAD_BUSY };
  if (res.status !== 202 || !data || typeof data.job_id !== "string") {
    return { error: typeof data?.detail === "string" ? data.detail : null };
  }

  const started = now();
  let delay = UPLOAD_POLL_FIRST_MS;
  let networkErrors = 0;
  onStatus("waiting to start");
  for (;;) {
    if (now() - started > maxWaitMs) return { error: UPLOAD_WAIT_EXCEEDED };
    await sleep(delay);
    delay = Math.min(Math.round(delay * 1.5), UPLOAD_POLL_MAX_MS);
    let job;
    try {
      const poll = await fetchImpl(`${backend}/upload/jobs/${encodeURIComponent(data.job_id)}`, {
        headers: { Authorization: `Bearer ${authToken || ""}` },
        cache: "no-store",
      });
      if (poll.status === 404) return { error: UPLOAD_JOB_LOST };
      if (!poll.ok) throw new Error(String(poll.status));
      job = await poll.json();
      networkErrors = 0;
    } catch (_) {
      networkErrors += 1;
      if (networkErrors >= MAX_POLL_NETWORK_ERRORS) return { error: UPLOAD_UNREACHABLE };
      continue;
    }
    if (job.status === "completed" || job.status === "failed") {
      const envelope = job.result;
      if (envelope && Array.isArray(envelope.results)) return { result: envelope.results[0] || null };
      return { error: typeof envelope?.error_message === "string" ? envelope.error_message : null };
    }
    onStatus(describeJob(job));
  }
}
