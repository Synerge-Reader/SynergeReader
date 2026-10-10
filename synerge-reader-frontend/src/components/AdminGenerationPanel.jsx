import { useCallback, useEffect, useRef, useState } from "react";

const DEFAULT_BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

// Fixed wording for the coordinator's fixed outcome categories. Anything not
// listed is shown as a generic failure rather than echoed.
const REASONS = {
  admin_abort: "aborted by an administrator",
  drain_timeout: "running work did not finish in time",
  uncertain_work: "model work with an unknown outcome is holding capacity",
  residency_timeout: "the previous model stayed loaded",
  residency_unavailable: "the loaded-model list could not be read",
  previous_profile_unknown: "the previous model is not registered",
  target_identity_mismatch: "the selected model is not installed with its qualified identity",
  target_identity_unavailable: "the selected model's identity could not be read",
  target_unavailable: "the selected model is not available",
  selection_conflict: "the selection changed meanwhile",
  ownership_unavailable: "this server does not own generation",
  interrupted_switch: "an interrupted switch was aborted",
  begin_ambiguous: "the switch could not be confirmed; generation stays paused",
  commit_ambiguous: "the commit could not be confirmed; generation stays paused",
  abort_ambiguous: "the abort could not be confirmed; generation stays paused",
  stopped: "the server stopped",
  coordinator_error: "the switch failed",
};

const PHASES = {
  queued: "queued",
  verifying_target: "checking the selected model",
  draining: "waiting for running answers to finish",
  awaiting_residency: "waiting for the previous model to unload",
  committing: "committing",
  aborting: "aborting",
  interrupted: "interrupted by a restart — abort it to resume answering",
};

function outcomeText(last) {
  if (!last) return null;
  const reason = last.reason ? REASONS[last.reason] || "the switch failed" : null;
  if (last.outcome === "committed") return "Last switch committed.";
  if (last.outcome === "aborted") return `Last switch aborted${reason ? `: ${reason}` : ""}.`;
  return `Last switch failed${reason ? `: ${reason}` : ""}.`;
}

/**
 * Admin-only control for the one shared answer model. Everything it shows
 * comes from the backend's /admin/generation status; the active model is the
 * committed one, never the one just requested. Authorization is enforced by
 * the backend on every call (the token travels in the Authorization header);
 * a non-admin sees nothing here.
 */
export default function AdminGenerationPanel({ backendUrl = DEFAULT_BACKEND, authToken, pollMs = 3000 }) {
  const [status, setStatus] = useState(null);
  const [state, setState] = useState("loading"); // loading | ready | forbidden | unavailable
  const [choice, setChoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const mounted = useRef(true);

  const headers = useCallback(
    (extra = {}) => ({ Authorization: `Bearer ${authToken}`, ...extra }),
    [authToken],
  );

  const refresh = useCallback(async () => {
    if (!authToken) return;
    try {
      const res = await fetch(`${backendUrl}/admin/generation`, { headers: headers(), cache: "no-store" });
      if (!mounted.current) return;
      if (res.status === 401 || res.status === 403) {
        setState("forbidden");
        return;
      }
      if (!res.ok) throw new Error("status unavailable");
      const payload = await res.json();
      if (!mounted.current) return;
      setStatus(payload);
      setState("ready");
    } catch {
      if (mounted.current) setState("unavailable");
    }
  }, [authToken, backendUrl, headers]);

  useEffect(() => {
    mounted.current = true;
    refresh();
    return () => { mounted.current = false; };
  }, [refresh]);

  const pending = Boolean(status && status.switch);

  useEffect(() => {
    if (!pending) return undefined;
    const timer = setInterval(refresh, pollMs);
    return () => clearInterval(timer);
  }, [pending, pollMs, refresh]);

  const send = async (path, body) => {
    setBusy(true);
    setNotice(null);
    try {
      const res = await fetch(`${backendUrl}${path}`, {
        method: "POST",
        headers: headers(body ? { "Content-Type": "application/json" } : {}),
        body: body ? JSON.stringify(body) : undefined,
      });
      let detail = null;
      try {
        const payload = await res.json();
        detail = typeof payload?.detail === "string" ? payload.detail : null;
      } catch {
        detail = null;
      }
      if (!res.ok) setNotice(detail || `Request failed (${res.status}).`);
    } catch {
      setNotice("The server could not be reached.");
    } finally {
      if (mounted.current) setBusy(false);
      await refresh();
    }
  };

  if (!authToken || state === "forbidden") return null;

  const active = status?.active;
  const choices = status?.choices || [];
  const eligible = choices.filter(c => c.eligible && !c.active);
  const selected = choice && eligible.some(c => c.profile_id === choice) ? choice : "";
  const capacity = status?.capacity;

  return (
    <section data-testid="admin-generation-panel" aria-label="Answer model" style={{
      background: "#fff", border: "1px solid #e5e7eb", borderRadius: "10px", padding: "14px 16px", marginBottom: "16px",
    }}>
      <div style={{ fontSize: "13px", fontWeight: 600, color: "#111827", marginBottom: "8px" }}>Answer model</div>
      {state === "loading" && <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>Checking…</div>}
      {state === "unavailable" && <div style={{ fontSize: "11.5px", color: "#b45309" }}>Model status is unavailable.</div>}
      {state === "ready" && status && (
        <div style={{ fontSize: "12px", color: "#374151" }}>
          <div data-testid="active-model" style={{ marginBottom: "6px" }}>
            Active: {active ? `${active.display_name || active.profile_id} (${active.model || "unregistered"})` : "none"}
            {active ? <span style={{ color: "#9ca3af" }}> · epoch {active.epoch}</span> : null}
          </div>
          <div style={{ color: "#6b7280", marginBottom: "6px" }}>
            Generation: {status.admission} · ownership: {status.ownership}
            {capacity && (capacity.unresolved_answers || capacity.unresolved_helpers)
              ? <span style={{ color: "#b45309" }}> · unresolved model work: {capacity.unresolved_answers + capacity.unresolved_helpers}</span>
              : null}
          </div>
          {pending && (
            <div data-testid="switch-pending" role="status" style={{ color: "#1d4ed8", marginBottom: "6px" }}>
              Switch pending{status.switch.target_profile_id ? ` to ${status.switch.target_profile_id}` : ""}:{" "}
              {PHASES[status.switch.phase] || status.switch.phase}
            </div>
          )}
          {!pending && outcomeText(status.last_switch) && (
            <div data-testid="switch-outcome" style={{ color: "#6b7280", marginBottom: "6px" }}>
              {outcomeText(status.last_switch)}
            </div>
          )}
          <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
            <label style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              <span>Switch to</span>
              <select
                aria-label="Model to switch to"
                value={selected}
                disabled={pending || busy}
                onChange={e => setChoice(e.target.value)}
                style={{ fontSize: "12px", padding: "3px 6px" }}
              >
                <option value="">Select a model</option>
                {choices.filter(c => !c.active).map(c => (
                  <option key={c.profile_id} value={c.profile_id} disabled={!c.eligible}>
                    {c.display_name} ({c.model}){c.eligible ? "" : " — not qualified"}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              disabled={pending || busy || !selected}
              onClick={() => send("/admin/generation/select", { profile_id: selected })}
              style={{ fontSize: "12px", padding: "4px 10px" }}
            >
              Switch model
            </button>
            {pending && (
              <button type="button" disabled={busy} onClick={() => send("/admin/generation/abort")}
                style={{ fontSize: "12px", padding: "4px 10px" }}>
                Abort switch
              </button>
            )}
          </div>
          {notice && <div role="alert" style={{ color: "#b91c1c", marginTop: "6px" }}>{notice}</div>}
        </div>
      )}
    </section>
  );
}
