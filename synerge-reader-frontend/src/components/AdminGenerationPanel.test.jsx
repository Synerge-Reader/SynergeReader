import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import AdminGenerationPanel from "./AdminGenerationPanel";

// Synthetic status payloads shaped exactly as the backend's /admin/generation sends them.
const CHOICES = [
  { profile_id: "llama_control", display_name: "Llama 3.1 8B", model: "llama3.1:8b", eligible: true, active: true },
  { profile_id: "gemma_candidate", display_name: "Gemma 4 31B", model: "gemma4:31b", eligible: true, active: false },
];
const CAPACITY = { answers_running: 0, helpers_running: 0, unresolved_answers: 0, unresolved_helpers: 0,
  answer_waiters: 0, total_slots: 2, answer_slots: 2, helper_slots: 1 };

function status(overrides = {}) {
  return {
    ownership: "owner",
    admission: "open",
    active: { profile_id: "llama_control", display_name: "Llama 3.1 8B", model: "llama3.1:8b", epoch: 1, registered: true },
    switch: null,
    last_switch: null,
    choices: CHOICES,
    capacity: CAPACITY,
    settings_provisional: true,
    ...overrides,
  };
}

const PENDING = status({
  admission: "paused",
  switch: { switch_id: "s-1", target_profile_id: "gemma_candidate", phase: "draining", abort_requested: false,
    origin: "this_process" },
});
const COMMITTED = status({
  active: { profile_id: "gemma_candidate", display_name: "Gemma 4 31B", model: "gemma4:31b", epoch: 3, registered: true },
  last_switch: { switch_id: "s-1", target_profile_id: "gemma_candidate", outcome: "committed", reason: null },
  choices: CHOICES.map(c => ({ ...c, active: c.profile_id === "gemma_candidate" })),
});

function reply(body, status = 200) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) });
}

// A scripted backend: each GET /admin/generation returns the next queued status
// (the last one repeats); POSTs return their scripted reply.
function backend({ statuses, post = { body: { switch_id: "s-1" }, status: 202 } }) {
  const queue = [...statuses];
  global.fetch = jest.fn((url, init = {}) => {
    if (url.endsWith("/admin/generation") && !init.method) {
      const next = queue.length > 1 ? queue.shift() : queue[0];
      return reply(next.body ?? next, next.status ?? 200);
    }
    return reply(post.body, post.status);
  });
}

afterEach(() => {
  delete global.fetch;
});

function renderPanel(props = {}) {
  return render(<AdminGenerationPanel backendUrl="https://example.test/api" authToken="tok-admin" pollMs={10} {...props} />);
}

test("shows the committed active model and asks with a bearer header, never a query token", async () => {
  backend({ statuses: [status()] });
  renderPanel();
  expect(await screen.findByTestId("active-model")).toHaveTextContent("Active: Llama 3.1 8B (llama3.1:8b)");
  const [url, init] = global.fetch.mock.calls[0];
  expect(url).toBe("https://example.test/api/admin/generation");
  expect(init.headers).toEqual({ Authorization: "Bearer tok-admin" });
  expect(url).not.toMatch(/token=/);
});

test("a non-admin sees no control at all", async () => {
  backend({ statuses: [{ body: { detail: "Forbidden: Admin access required" }, status: 403 }] });
  const { container } = renderPanel({ authToken: "tok-alice" });
  await waitFor(() => expect(global.fetch).toHaveBeenCalled());
  await waitFor(() => expect(container).toBeEmptyDOMElement());
});

test("ineligible models cannot be chosen", async () => {
  backend({ statuses: [status({ choices: [CHOICES[0], { ...CHOICES[1], eligible: false }] })] });
  renderPanel();
  const option = await screen.findByRole("option", { name: /Gemma 4 31B/ });
  expect(option).toBeDisabled();
  expect(option).toHaveTextContent("not qualified");
  expect(screen.getByRole("button", { name: "Switch model" })).toBeDisabled();
});

test("a switch shows pending, keeps the old model active, and shows the new one only after commit", async () => {
  backend({ statuses: [status(), PENDING, PENDING, COMMITTED] });
  renderPanel();
  fireEvent.change(await screen.findByRole("combobox", { name: "Model to switch to" }), {
    target: { value: "gemma_candidate" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Switch model" }));

  expect(await screen.findByTestId("switch-pending")).toHaveTextContent("waiting for running answers to finish");
  expect(screen.getByTestId("active-model")).toHaveTextContent("Llama 3.1 8B");
  expect(screen.getByRole("button", { name: "Switch model" })).toBeDisabled();
  expect(screen.getByRole("combobox", { name: "Model to switch to" })).toBeDisabled();

  const post = global.fetch.mock.calls.find(([, init]) => init && init.method === "POST");
  expect(post[0]).toBe("https://example.test/api/admin/generation/select");
  expect(post[1].headers).toEqual({ Authorization: "Bearer tok-admin", "Content-Type": "application/json" });
  expect(JSON.parse(post[1].body)).toEqual({ profile_id: "gemma_candidate" });

  await waitFor(() => expect(screen.getByTestId("active-model")).toHaveTextContent("Gemma 4 31B (gemma4:31b)"));
  expect(screen.getByTestId("switch-outcome")).toHaveTextContent("Last switch committed.");
  expect(screen.queryByTestId("switch-pending")).toBeNull();
});

test("a refused switch shows the server's reason and changes nothing", async () => {
  backend({
    statuses: [status()],
    post: { body: { detail: "A model switch is already in progress." }, status: 409 },
  });
  renderPanel();
  fireEvent.change(await screen.findByRole("combobox", { name: "Model to switch to" }), {
    target: { value: "gemma_candidate" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Switch model" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("A model switch is already in progress.");
  expect(screen.getByTestId("active-model")).toHaveTextContent("Llama 3.1 8B");
});

test("a pending switch can be aborted and the failure reason is shown in fixed words", async () => {
  const aborted = status({
    active: { ...status().active, epoch: 3 },
    last_switch: { switch_id: "s-1", target_profile_id: "gemma_candidate", outcome: "aborted", reason: "admin_abort" },
  });
  backend({ statuses: [PENDING, aborted], post: { body: { switch_id: "s-1" }, status: 202 } });
  renderPanel({ pollMs: 100000 });
  fireEvent.click(await screen.findByRole("button", { name: "Abort switch" }));
  expect(await screen.findByTestId("switch-outcome")).toHaveTextContent("Last switch aborted: aborted by an administrator.");
  const post = global.fetch.mock.calls.find(([, init]) => init && init.method === "POST");
  expect(post[0]).toBe("https://example.test/api/admin/generation/abort");
  expect(screen.getByTestId("active-model")).toHaveTextContent("Llama 3.1 8B");
});

test("a return to control waiting on an unreadable runtime says unloading cannot be confirmed", async () => {
  const waiting = status({
    admission: "paused",
    switch: { switch_id: "s-2", target_profile_id: "gemma_candidate", phase: "awaiting_residency_unverified",
      abort_requested: false, origin: "this_process" },
  });
  backend({ statuses: [waiting] });
  renderPanel({ pollMs: 100000 });
  expect(await screen.findByTestId("switch-pending")).toHaveTextContent(
    "the previous model's runtime cannot be read; waiting the full residency window before returning " +
    "(its unloading cannot be confirmed)");
  expect(screen.getByRole("button", { name: "Abort switch" })).toBeEnabled();
});

test("a commit with residency unverified is reported as such, never as a plain commit", async () => {
  const recovered = status({
    active: { profile_id: "gemma_candidate", display_name: "Gemma 4 31B", model: "gemma4:31b", epoch: 7, registered: true },
    last_switch: { switch_id: "s-2", target_profile_id: "gemma_candidate", outcome: "committed",
      reason: "residency_unverified" },
    choices: CHOICES.map(c => ({ ...c, active: c.profile_id === "gemma_candidate" })),
  });
  backend({ statuses: [recovered] });
  renderPanel({ pollMs: 100000 });
  const outcome = await screen.findByTestId("switch-outcome");
  expect(outcome).toHaveTextContent("Last switch committed with residency unverified");
  expect(outcome).toHaveTextContent("That does not confirm the previous model unloaded.");
  expect(outcome).not.toHaveTextContent(/^Last switch committed\.$/);
});

test("an unreadable status says so instead of guessing", async () => {
  global.fetch = jest.fn(() => Promise.reject(new TypeError("network down")));
  renderPanel();
  expect(await screen.findByText("Model status is unavailable.")).toBeInTheDocument();
  expect(screen.queryByTestId("active-model")).toBeNull();
});

// --- experimental profiles ---------------------------------------------------------

const MISTRAL_NOTE = "Residency 91.27% against the 99% target; 180.38 s load.";
const EXPERIMENTAL = [
  { profile_id: "qwen38_27b_experimental", display_name: "Qwen3.8 27B", model: "qwen3.8:27b", eligible: true,
    ineligible_reason: null, active: false, runtime: "compat_cuda121", evidence: "experimental", experimental: true,
    runtime_smoke_verdict: "PASS", review_note: null },
  { profile_id: "mistral_medium35_128b_experimental", display_name: "Mistral Medium 3.5 128B",
    model: "mistral-medium-3.5:128b", eligible: true, ineligible_reason: null, active: false,
    runtime: "compat_cuda121", evidence: "experimental", experimental: true,
    runtime_smoke_verdict: "REVIEW_REQUIRED", review_note: MISTRAL_NOTE },
];

test("ineligible experimental models say why, in fixed words", async () => {
  backend({ statuses: [status({
    experimental_enabled: false,
    choices: [CHOICES[0], ...EXPERIMENTAL.map(c => ({ ...c, eligible: false, ineligible_reason: "experimental_disabled" })),
      { ...EXPERIMENTAL[0], profile_id: "x_experimental", display_name: "Other", eligible: false,
        ineligible_reason: "runtime_not_configured" }],
  })] });
  renderPanel();
  const qwen = await screen.findByRole("option", { name: /Qwen3.8 27B/ });
  expect(qwen).toBeDisabled();
  expect(qwen).toHaveTextContent("experimental — experimental models are disabled");
  expect(screen.getByRole("option", { name: /Other/ })).toHaveTextContent("its runtime is not configured");
});

test("choosing an experimental model states its evidence honestly before the switch", async () => {
  backend({ statuses: [status({ experimental_enabled: true, choices: [...CHOICES, ...EXPERIMENTAL] })] });
  renderPanel();
  fireEvent.change(await screen.findByRole("combobox", { name: "Model to switch to" }), {
    target: { value: "mistral_medium35_128b_experimental" },
  });
  const details = await screen.findByTestId("selected-experimental-details");
  expect(details).toHaveTextContent("runtime smoke test only (review required)");
  expect(details).toHaveTextContent("Not quality-, capacity- or vision-qualified.");
  expect(details).toHaveTextContent("compat_cuda121 runtime, one answer or background job at a time");
  expect(details).toHaveTextContent("can take several minutes; a switch does not preload it");
  expect(screen.getByTestId("selected-experimental-details-note")).toHaveTextContent(MISTRAL_NOTE);
  expect(details).not.toHaveTextContent(/qualified for use|warmed|ready to answer/i);

  fireEvent.change(screen.getByRole("combobox", { name: "Model to switch to" }), {
    target: { value: "qwen38_27b_experimental" },
  });
  expect(screen.getByTestId("selected-experimental-details")).toHaveTextContent("(passed)");
  expect(screen.queryByTestId("selected-experimental-details-note")).toBeNull();
  fireEvent.change(screen.getByRole("combobox", { name: "Model to switch to" }), {
    target: { value: "gemma_candidate" },
  });
  expect(screen.queryByTestId("selected-experimental-details")).toBeNull();
});

test("an active experimental model is badged with its job limit, and an unservable one is flagged", async () => {
  const mistral = { ...EXPERIMENTAL[1], active: true, eligible: false, ineligible_reason: "experimental_disabled" };
  backend({ statuses: [status({
    experimental_enabled: false,
    active: { profile_id: mistral.profile_id, display_name: mistral.display_name, model: mistral.model, epoch: 6,
      registered: true, experimental: true, runtime: "compat_cuda121", eligible: false },
    choices: [{ ...CHOICES[0], active: false }, CHOICES[1], mistral],
    capacity: { ...CAPACITY, selection_job_limit: 1 },
  })] });
  renderPanel();
  expect(await screen.findByTestId("active-experimental")).toHaveTextContent("Experimental");
  expect(screen.getByTestId("active-ineligible")).toHaveTextContent(
    "This server will not answer with the active model (experimental models are disabled)");
  expect(screen.getByTestId("active-experimental-details-note")).toHaveTextContent(MISTRAL_NOTE);
  expect(screen.getByTestId("job-limit")).toHaveTextContent("job limit: 1 at a time");
  // The way back is the coordinated switch to an available model.
  const gemma = screen.getByRole("option", { name: /Gemma 4 31B/ });
  expect(gemma).not.toBeDisabled();
});

test("a control or qualified active model shows no experimental marking or job limit", async () => {
  backend({ statuses: [status()] });
  renderPanel();
  await screen.findByTestId("active-model");
  expect(screen.queryByTestId("active-experimental")).toBeNull();
  expect(screen.queryByTestId("active-ineligible")).toBeNull();
  expect(screen.queryByTestId("job-limit")).toBeNull();
});
