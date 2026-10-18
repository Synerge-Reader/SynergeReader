import { UPLOAD_BUSY, UPLOAD_JOB_LOST, UPLOAD_UNREACHABLE, UPLOAD_WAIT_EXCEEDED, uploadAndWait } from "./uploadJobs";

const BACKEND = "https://example.test/api";
const INDEXED = { status: "indexed", document_id: 42, filename: "scan.pdf", chunks_count: 3, warnings: ["Page 1 was read with OCR."] };

function reply(status, body) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) });
}

function scripted(responses) {
  const calls = [];
  const fetchImpl = jest.fn((url, init) => {
    calls.push({ url, init });
    const next = responses.shift();
    if (next instanceof Error) return Promise.reject(next);
    return reply(next[0], next[1]);
  });
  return { fetchImpl, calls };
}

function run(fetchImpl, extra = {}) {
  let clock = 0;
  return uploadAndWait({
    backend: BACKEND,
    file: new File(["%PDF-1.4"], "scan.pdf", { type: "application/pdf" }),
    authToken: "tok-alice",
    extraction: "enhanced",
    fetchImpl,
    sleep: ms => { clock += ms; return Promise.resolve(); },
    now: () => clock,
    ...extra,
  });
}

test("submits for background processing and returns the finished job's first result", async () => {
  const statuses = [];
  const { fetchImpl, calls } = scripted([
    [202, { job_id: "job-1", status: "queued", status_url: "/upload/jobs/job-1" }],
    [200, { job_id: "job-1", status: "queued", queue_position: 2 }],
    [200, { job_id: "job-1", status: "running" }],
    [200, { job_id: "job-1", status: "completed", http_status: 200, result: { results: [INDEXED] } }],
  ]);
  const out = await run(fetchImpl, { onStatus: s => statuses.push(s) });
  expect(out).toEqual({ result: INDEXED });
  const form = calls[0].init.body;
  expect(calls[0].url).toBe(`${BACKEND}/upload`);
  expect(form.get("processing")).toBe("background");
  expect(form.get("extraction")).toBe("enhanced");
  expect(form.get("auth_token")).toBe("tok-alice");
  expect(calls[1].url).toBe(`${BACKEND}/upload/jobs/job-1`);
  expect(calls[1].init.headers.Authorization).toBe("Bearer tok-alice");
  expect(statuses).toEqual(["uploading", "waiting to start", "waiting (position 2)", "reading pages"]);
});

test("a failed job surfaces its non-indexed result instead of success", async () => {
  const failed = { status: "failed", error_message: "The PDF is effectively empty." };
  const { fetchImpl } = scripted([
    [202, { job_id: "job-2" }],
    [200, { job_id: "job-2", status: "completed", http_status: 422, result: { results: [failed] } }],
  ]);
  expect(await run(fetchImpl)).toEqual({ result: failed });
});

test("a job that disappears is reported as lost", async () => {
  const { fetchImpl } = scripted([[202, { job_id: "job-3" }], [404, { detail: "Upload job not found or expired" }]]);
  expect(await run(fetchImpl)).toEqual({ error: UPLOAD_JOB_LOST });
});

test("a busy server answers with a retry message", async () => {
  const { fetchImpl } = scripted([[429, { detail: "busy" }]]);
  expect(await run(fetchImpl)).toEqual({ error: UPLOAD_BUSY });
});

test("a synchronous envelope is used directly", async () => {
  const { fetchImpl } = scripted([[200, { results: [INDEXED] }]]);
  expect(await run(fetchImpl)).toEqual({ result: INDEXED });
});

test("repeated polling failures stop with an unreachable message", async () => {
  const { fetchImpl } = scripted([[202, { job_id: "job-4" }], ...Array.from({ length: 5 }, () => new Error("offline"))]);
  expect(await run(fetchImpl)).toEqual({ error: UPLOAD_UNREACHABLE });
});

test("waiting is bounded", async () => {
  const { fetchImpl } = scripted([[202, { job_id: "job-5" }], ...Array.from({ length: 50 }, () => [200, { status: "running" }])]);
  expect(await run(fetchImpl, { maxWaitMs: 20000 })).toEqual({ error: UPLOAD_WAIT_EXCEEDED });
});
