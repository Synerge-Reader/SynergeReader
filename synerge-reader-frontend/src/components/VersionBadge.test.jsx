import { render, screen, waitFor } from "@testing-library/react";
import VersionBadge, { VERSION_UNAVAILABLE, versionLabel } from "./VersionBadge";

// Synthetic example identities, shaped exactly as the backend's /version sends them.
const KNOWN = {
  app_version: "0.1.0",
  build_sha: "0123456789abcdef0123456789abcdef01234567",
  build_sha_short: "0123456",
  build_known: true,
  display_version: "v0.1.0 · 0123456",
};
const UNKNOWN = {
  app_version: "0.1.0",
  build_sha: null,
  build_sha_short: null,
  build_known: false,
  display_version: "v0.1.0 · build unknown",
};

function respondWith(body, { ok = true, status = 200 } = {}) {
  global.fetch = jest.fn(() => Promise.resolve({ ok, status, json: () => Promise.resolve(body) }));
}

afterEach(() => {
  delete global.fetch;
});

async function renderedVersion(props = {}) {
  render(<VersionBadge backendUrl="https://example.test/api" {...props} />);
  return screen.findByTestId("app-version");
}

test("a known build shows the version and the short commit from the server", async () => {
  respondWith(KNOWN);
  const badge = await renderedVersion();
  expect(badge).toHaveTextContent("v0.1.0 · 0123456");
  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledWith("https://example.test/api/version", { cache: "no-store" });
});

test("an unknown build says so instead of guessing", async () => {
  respondWith(UNKNOWN);
  const badge = await renderedVersion();
  expect(badge).toHaveTextContent("v0.1.0 · build unknown");
});

test("a failed request reads as unavailable", async () => {
  global.fetch = jest.fn(() => Promise.reject(new TypeError("network down")));
  const badge = await renderedVersion();
  expect(badge).toHaveTextContent(VERSION_UNAVAILABLE);
  expect(badge).not.toHaveTextContent("v0.1.0");
});

test("an error status reads as unavailable even if it has a body", async () => {
  respondWith(KNOWN, { ok: false, status: 503 });
  const badge = await renderedVersion();
  expect(badge).toHaveTextContent(VERSION_UNAVAILABLE);
});

test("a malformed response reads as unavailable", async () => {
  respondWith({ app_version: "0.1.0", build_known: true, build_sha_short: "<b>nope</b>" });
  const badge = await renderedVersion();
  expect(badge).toHaveTextContent(VERSION_UNAVAILABLE);
});

test("the frontend's own build or git state is never substituted", async () => {
  const saved = { ...process.env };
  process.env.REACT_APP_BUILD_SHA = "fedcba9876543210fedcba9876543210fedcba98";
  process.env.REACT_APP_GIT_SHA = "fedcba9876543210fedcba9876543210fedcba98";
  process.env.REACT_APP_VERSION = "9.9.9";
  try {
    global.fetch = jest.fn(() => Promise.reject(new Error("offline")));
    const badge = await renderedVersion();
    expect(badge).toHaveTextContent(VERSION_UNAVAILABLE);
    expect(badge.textContent).not.toMatch(/fedcba|9\.9\.9/);
  } finally {
    process.env = saved;
  }
});

test("the version is exposed to assistive technology as labelled text", async () => {
  respondWith(KNOWN);
  const badge = await renderedVersion();
  expect(badge).not.toHaveAttribute("aria-hidden");
  expect(badge.textContent).toBe("Application version: v0.1.0 · 0123456");
  expect(screen.getByText("Application version:")).toBeInTheDocument();
});

test("nothing internal is shown, even if the server sends extra fields", async () => {
  respondWith({
    ...KNOWN,
    models: ["llama3.1:8b", "gemma4:31b"],
    profiles: ["llama_control", "gemma_candidate"],
    ollama_url: "http://172.18.0.1:11434",
    db: "postgresql://user@synergereader-db:5432/synergereader",
    path: "/mnt/2025-legalread/repo",
  });
  const badge = await renderedVersion();
  expect(badge.textContent).toBe("Application version: v0.1.0 · 0123456");
  for (const internal of [/llama/i, /gemma/i, /qwen/i, /profile/i, /11434/, /postgres/i, /\/mnt\//, /0123456789abcdef/]) {
    expect(badge.textContent).not.toMatch(internal);
  }
});

test("startup is not blocked while the server has not answered", () => {
  global.fetch = jest.fn(() => new Promise(() => {}));
  const { container } = render(
    <div>
      <p>app content</p>
      <VersionBadge backendUrl="https://example.test/api" />
    </div>
  );
  expect(screen.getByText("app content")).toBeInTheDocument();
  expect(screen.queryByTestId("app-version")).not.toBeInTheDocument();
  expect(container).not.toHaveTextContent(VERSION_UNAVAILABLE);
});

test("the default backend URL follows the app's REACT_APP_BACKEND_URL convention", async () => {
  respondWith(KNOWN);
  render(<VersionBadge />);
  await screen.findByTestId("app-version");
  const expectedBase = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
  expect(global.fetch).toHaveBeenCalledWith(`${expectedBase}/version`, { cache: "no-store" });
});

test("a late response after unmount does not throw", async () => {
  let resolve;
  global.fetch = jest.fn(() => new Promise(r => { resolve = r; }));
  const { unmount } = render(<VersionBadge backendUrl="https://example.test/api" />);
  unmount();
  resolve({ ok: true, json: () => Promise.resolve(KNOWN) });
  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
});

describe("versionLabel", () => {
  test.each([
    [KNOWN, "v0.1.0 · 0123456"],
    [UNKNOWN, "v0.1.0 · build unknown"],
    [{ ...KNOWN, build_known: "true" }, VERSION_UNAVAILABLE],
    [{ ...KNOWN, app_version: "v0.1.0" }, VERSION_UNAVAILABLE],
    [{ ...UNKNOWN, build_known: undefined }, VERSION_UNAVAILABLE],
    [null, VERSION_UNAVAILABLE],
    ["v0.1.0", VERSION_UNAVAILABLE],
  ])("%j -> %s", (payload, expected) => {
    expect(versionLabel(payload)).toBe(expected);
  });
});
