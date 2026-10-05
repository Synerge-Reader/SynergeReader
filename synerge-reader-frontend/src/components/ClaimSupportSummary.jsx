import { useState } from "react";

// What happened to one claim, in the reader's language. The backend's reason
// is what separates "we checked and it is not supported" from "we never got to
// check it" — collapsing those into one "not verified" count is what made the
// old pills unreadable.
export const CLAIM_STATE_LABEL = {
  supported:           { label: "Supported",                        color: "#15803d", background: "#dcfce7", border: "#86efac" },
  partially_supported: { label: "Partly supported",                 color: "#b45309", background: "#fef3c7", border: "#fcd34d" },
  no_citation:         { label: "Not supported by a cited document", color: "#b91c1c", background: "#fee2e2", border: "#fca5a5" },
  invalid_citation:    { label: "Citation could not be resolved",    color: "#b91c1c", background: "#fee2e2", border: "#fca5a5" },
  // The claim did cite a source; that source just does not state it. Saying
  // "not supported by a cited document" would misdescribe what went wrong.
  evidence_mismatch:   { label: "Cited source does not state this",  color: "#b91c1c", background: "#fee2e2", border: "#fca5a5" },
  // The verifier read the cited source and judged the claim unsupported: a
  // completed check of a cited claim, not a missing citation.
  checked_unsupported: { label: "Cited source does not support this", color: "#b91c1c", background: "#fee2e2", border: "#fca5a5" },
  unsupported_other:   { label: "Not supported",                     color: "#b91c1c", background: "#fee2e2", border: "#fca5a5" },
  not_checked:         { label: "Not checked",                       color: "#475569", background: "#f1f5f9", border: "#cbd5e1" },
  check_unavailable:   { label: "Check unavailable",                 color: "#475569", background: "#f1f5f9", border: "#cbd5e1" },
};

export function claimStateKey(claim) {
  if (claim.status === "supported") return "supported";
  if (claim.status === "partially_supported") return "partially_supported";
  if (claim.status === "unsupported") {
    if (claim.reason === "invalid_citation") return "invalid_citation";
    if (claim.reason === "evidence_terms_missing") return "evidence_mismatch";
    if (claim.reason === "no_citation") return "no_citation";
    if (claim.reason === "verified") return "checked_unsupported";
    return "unsupported_other";
  }
  // unverified: the reason says whether we ran out of budget or the verifier
  // itself failed. Neither means the claim is wrong.
  return (claim.reason === "verifier_unavailable" || claim.reason === "verifier_malformed")
    ? "check_unavailable"
    : "not_checked";
}

export const SUPPORT_HEADLINE = {
  supported:   "Supported by your documents",
  partial:     "Partly supported by your documents",
  unsupported: "Parts of this answer are not supported by your documents",
  mixed:       "Some claims in this answer need checking",
  incomplete:  "Citation check incomplete",
  unavailable: "Citation check unavailable",
};

const HEADLINE_TONE = {
  supported:   "#15803d",
  partial:     "#b45309",
  unsupported: "#b91c1c",
  mixed:       "#b45309",
  incomplete:  "#475569",
  unavailable: "#475569",
};

const UNSUPPORTED_KEYS = ["no_citation", "invalid_citation", "evidence_mismatch", "checked_unsupported", "unsupported_other"];

const KNOWN_STATUSES = ["supported", "partially_supported", "unsupported", "unverified"];
const KNOWN_REASONS = [
  "no_citation", "invalid_citation", "verifier_unavailable", "verifier_malformed",
  "not_verified", "verified", "evidence_terms_missing",
];
// A verdict was reached: the verifier answered (and the deterministic floor
// may have lowered it). Unsupported verdicts are completed checks too.
const CHECKED_REASONS = ["verified", "evidence_terms_missing"];

// What was checked, from the claims the backend reported. Counts are "known"
// only when every claim carries a status and reason this client understands;
// otherwise nothing is counted and the UI says the coverage is unknown.
export function claimCoverage(claims) {
  const list = Array.isArray(claims) ? claims : [];
  const known = list.every(c =>
    c && KNOWN_STATUSES.includes(c.status) && KNOWN_REASONS.includes(c.reason) && Array.isArray(c.citation_ids)
  );
  // Eligible: the claim cited at least one supplied source, so the verifier
  // could check it. The backend caps how many it checks.
  const eligible = list.filter(c => Array.isArray(c?.citation_ids) && c.citation_ids.length > 0);
  const checked = eligible.filter(c => CHECKED_REASONS.includes(c.reason));
  return {
    known,
    total: list.length,
    eligible: eligible.length,
    checked: checked.length,
    unchecked: eligible.length - checked.length,
    uncited: list.length - eligible.length,
    supported: checked.filter(c => c.status === "supported").length,
    partial: checked.filter(c => c.status === "partially_supported").length,
    unsupported: checked.filter(c => c.status === "unsupported").length,
  };
}

export function coverageText(coverage) {
  if (!coverage.known) return "Check coverage unknown";
  if (coverage.total === 0) return "No claims were checked";
  if (coverage.eligible === 0) return "No cited claims to check";
  const plural = coverage.eligible === 1 ? "" : "s";
  let text = `${coverage.checked} of ${coverage.eligible} cited claim${plural} checked`;
  const verdicts = [
    coverage.supported && `${coverage.supported} supported`,
    coverage.partial && `${coverage.partial} partly supported`,
    coverage.unsupported && `${coverage.unsupported} not supported`,
  ].filter(Boolean);
  if (verdicts.length) text += ` (${verdicts.join(", ")})`;
  const parts = [text];
  if (coverage.unchecked > 0) parts.push(`${coverage.unchecked} not checked`);
  if (coverage.uncited > 0) parts.push(`${coverage.uncited} without a usable citation`);
  return parts.join(" · ");
}

// The headline is a summary, not a verdict on the weakest sentence. An answer
// whose substantive claims are all cited and supported should not be shown as
// wholly unsupported because it opened with a framing sentence like "IRCoT and
// Step-Back Prompting differ in their approach" — that sentence asserts
// nothing about the documents, so nothing can cite it.
//
// The rules that do not move: an answer with no supported claim at all stays
// red, and green means every claim was checked and fully supported. A partly
// supported claim, an unchecked claim, or unknown coverage is never green.
export function supportHeadlineKey(claims) {
  const keys = claims.map(claimStateKey);
  const unsupported = keys.filter(k => UNSUPPORTED_KEYS.includes(k)).length;
  const supported = keys.filter(k => k === "supported" || k === "partially_supported").length;
  if (unsupported > 0 && supported === 0) return "unsupported";
  if (unsupported > 0) return "mixed";
  if (keys.some(k => k === "check_unavailable")) return "unavailable";
  if (keys.some(k => k === "not_checked")) return "incomplete";
  if (!claimCoverage(claims).known) return "incomplete";
  if (keys.some(k => k === "partially_supported")) return "partial";
  return "supported";
}

// Claim text still carries raw [Cn] registry ids. The caller passes the same
// rewriter it uses for the answer body; this fallback only removes the ids so
// an internal id can never be printed.
function PlainClaimText({ text }) {
  return <>{(text || "").replace(/\s*\[C\d+\]/g, "")}</>;
}

export function ClaimSupportSummary({ claims, citations, onOpen, ClaimText = PlainClaimText }) {
  const [open, setOpen] = useState(false);
  // No verification event yet (or none for this kind of answer): say nothing.
  if (!Array.isArray(claims)) return null;
  // A verification that produced no claims checked nothing; it is not a pass.
  const headlineKey = claims.length ? supportHeadlineKey(claims) : "unavailable";
  const tone = HEADLINE_TONE[headlineKey];
  return (
    <div style={{ marginTop: "8px" }}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        style={{
          display: "inline-flex", alignItems: "center", gap: "6px",
          background: "none", border: "none", padding: 0, cursor: "pointer",
          fontSize: "11.5px", fontWeight: 700, color: tone,
        }}
      >
        {SUPPORT_HEADLINE[headlineKey]}
        {claims.length > 0 && (
          <span style={{ fontSize: "10px", color: "#9ca3af" }}>{open ? "▲ hide details" : "▼ details"}</span>
        )}
      </button>
      <div style={{ fontSize: "10.5px", color: "#6b7280", marginTop: "2px" }}>
        {coverageText(claimCoverage(claims))}
      </div>
      {open && (
        <div style={{ marginTop: "6px", display: "flex", flexDirection: "column", gap: "5px" }}>
          {claims.map((claim, i) => {
            const state = CLAIM_STATE_LABEL[claimStateKey(claim)];
            return (
              <div key={i} style={{ display: "flex", gap: "7px", alignItems: "flex-start" }}>
                <span style={{
                  flexShrink: 0, padding: "1px 7px", borderRadius: "10px", fontSize: "10px",
                  fontWeight: 700, color: state.color, background: state.background,
                  border: `1px solid ${state.border}`,
                }}>{state.label}</span>
                <span style={{ fontSize: "11.5px", color: "#4b5563", lineHeight: "1.5" }}>
                  {/* The claim text is what the model wrote, so it still
                      carries raw [Cn] registry ids. It goes through the same
                      rewriter as the answer body so the reader sees compact
                      public numbers here too, never an internal id. */}
                  <ClaimText text={claim.text} citations={citations} pending={false} onOpen={onOpen} />
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
