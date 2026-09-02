import { useState, useRef, useEffect, useCallback } from "react";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import { GlobalWorkerOptions } from "pdfjs-dist/build/pdf";
import { renderAsync } from "docx-preview";
import UserAuth from "./components/UserAuth/UserAuth";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();

const BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

// UI font stack — used for chrome/chat. Document content panes keep their own serif styling.
const UI_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
const TASK_MODES = [
  { id: "research",   label: "Research & Q&A",    model: "llama3.1:8b",  color: "#0891b2" },
  { id: "argument",   label: "Argument Generator", model: "qwen3:latest", color: "#7c3aed" },
  { id: "precedents", label: "Related Precedents", model: "qwen3:latest", color: "#7c3aed" },
  { id: "risk",       label: "Risk Analysis",      model: "qwen3:latest", color: "#dc2626" },
  { id: "clause",     label: "Clause Extractor",   model: "llama3.1:8b",  color: "#0891b2" },
  { id: "summarize",  label: "Summarize",          model: "qwen3:latest", color: "#7c3aed" },
];

const MODEL_LABEL = {
  "llama3.1:8b":  "Llama 3.1 8B",
  "qwen3:latest": "Qwen3",
};

const TASK_PROMPTS = {
  argument:   "You are a legal research assistant. Structure every argument in IRAC format: Issue, Rule, Application, Conclusion. Cite specific page numbers for every claim. Flag weaknesses opposing counsel might exploit.",
  risk:       "You are a legal risk analyst. Identify ambiguous language, missing standard clauses, unfavorable terms, and jurisdiction risks. Rate each risk High/Medium/Low with page citations.",
  clause:     "You are a contract analysis assistant. Extract the requested clause type precisely. Return: exact clause text, plain English explanation, and risk level (Low/Medium/High). If missing, state explicitly.",
  summarize:  "You are a legal document analyst. Provide a structured summary covering: parties involved, key dates, main obligations, and notable clauses or findings.",
  precedents: "You are a legal research assistant. Identify and explain relevant legal precedents from the document. Note applicable jurisdictions and how they relate to the case.",
  research:   "You are a document assistant. Answer only from the provided context when possible. If insufficient, say what is missing instead of guessing.",
};

const CLAUSE_TYPES = [
  "Termination", "Indemnification", "Confidentiality", "Non-Compete",
  "Limitation of Liability", "Governing Law", "Payment Terms",
  "Force Majeure", "Dispute Resolution", "Assignment",
];


// ─────────────────────────────────────────────────────────────────────────────
// ICONS — inline SVG, outline style
// ─────────────────────────────────────────────────────────────────────────────
const iconBase = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };

function IconPaperclip(props) {
  return (<svg {...iconBase} {...props}><path d="M21.44 11.05l-9.19 9.19a5.5 5.5 0 01-7.78-7.78l9.19-9.19a3.5 3.5 0 014.95 4.95l-9.2 9.19a1.5 1.5 0 01-2.12-2.12l8.49-8.48" /></svg>);
}
function IconArrowUp(props) {
  return (<svg {...iconBase} {...props}><path d="M12 19V5M5 12l7-7 7 7" /></svg>);
}
function IconPlus(props) {
  return (<svg {...iconBase} {...props}><path d="M12 5v14M5 12h14" /></svg>);
}
function IconPanel(props) {
  return (<svg {...iconBase} {...props}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></svg>);
}
function IconBook(props) {
  return (<svg {...iconBase} {...props}><path d="M4 19.5A2.5 2.5 0 016.5 17H20V4H6.5A2.5 2.5 0 004 6.5v13z" /><path d="M20 17v3H6.5A2.5 2.5 0 014 17.5" /></svg>);
}
function IconDatabase(props) {
  return (<svg {...iconBase} {...props}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" /><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" /></svg>);
}
function IconDownload(props) {
  return (<svg {...iconBase} {...props}><path d="M12 3v12M7 10l5 5 5-5" /><path d="M4 19h16" /></svg>);
}
function IconX(props) {
  return (<svg {...iconBase} {...props}><path d="M18 6L6 18M6 6l12 12" /></svg>);
}
function IconTrash(props) {
  return (<svg {...iconBase} {...props}><path d="M3 6h18" /><path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2" /><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" /></svg>);
}
function IconThumbUp(props) {
  return (<svg {...iconBase} {...props}><path d="M7 22V11M2 13v7a2 2 0 002 2h11.5a2 2 0 002-1.5l1.8-7A2 2 0 0017.4 11H13V5a2 2 0 00-2-2L7 11" /></svg>);
}
function IconThumbDown(props) {
  return (<svg {...iconBase} {...props}><path d="M17 2v11M22 11V4a2 2 0 00-2-2H8.5a2 2 0 00-2 1.5l-1.8 7A2 2 0 006.6 13H11v6a2 2 0 002 2l4-8" /></svg>);
}
function IconFile(props) {
  return (<svg {...iconBase} {...props}><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><path d="M14 2v6h6" /></svg>);
}
function IconLayers(props) {
  return (<svg {...iconBase} {...props}><path d="M12 2l9 5-9 5-9-5 9-5z" /><path d="M3 12l9 5 9-5" /><path d="M3 17l9 5 9-5" /></svg>);
}
function IconShield(props) {
  return (<svg {...iconBase} {...props}><path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-4z" /></svg>);
}
function IconGrid(props) {
  return (<svg {...iconBase} {...props}><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>);
}
function IconUser(props) {
  return (<svg {...iconBase} {...props}><circle cx="12" cy="8" r="4" /><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8" /></svg>);
}
function IconLogout(props) {
  return (<svg {...iconBase} {...props}><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" /><path d="M16 17l5-5-5-5" /><path d="M21 12H9" /></svg>);
}
function IconLock(props) {
  return (<svg {...iconBase} {...props}><rect x="4" y="11" width="16" height="10" rx="2" /><path d="M8 11V7a4 4 0 018 0v4" /></svg>);
}
function IconBan(props) {
  return (<svg {...iconBase} {...props}><circle cx="12" cy="12" r="9" /><path d="M5.5 5.5l13 13" /></svg>);
}
function IconCheckCircle(props) {
  return (<svg {...iconBase} {...props}><circle cx="12" cy="12" r="9" /><path d="M8.5 12.5l2.5 2.5 5-5" /></svg>);
}
function IconChevronRight(props) {
  return (<svg {...iconBase} {...props}><path d="M9 6l6 6-6 6" /></svg>);
}
function IconRefresh(props) {
  return (<svg {...iconBase} {...props}><path d="M21 12a9 9 0 10-3 6.7" /><path d="M21 5v6h-6" /></svg>);
}
function IconAlertTriangle(props) {
  return (<svg {...iconBase} {...props}><path d="M12 3l10 18H2L12 3z" /><path d="M12 10v4" /><circle cx="12" cy="17.3" r="0.4" fill="currentColor" stroke="none" /></svg>);
}
function IconSearch(props) {
  return (<svg {...iconBase} {...props}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></svg>);
}
function IconChevronDown(props) {
  return (<svg {...iconBase} {...props}><path d="M6 9l6 6 6-6" /></svg>);
}
function IconTag(props) {
  return (<svg {...iconBase} {...props}><path d="M20.6 12.6L12 4H4v8l8.6 8.6a2 2 0 002.8 0l5.2-5.2a2 2 0 000-2.8z" /><circle cx="8" cy="8" r="1.4" fill="currentColor" stroke="none" /></svg>);
}
function IconListChecks(props) {
  return (<svg {...iconBase} {...props}><path d="M4 6l1.5 1.5L8 5" /><path d="M4 12l1.5 1.5L8 11" /><path d="M4 18l1.5 1.5L8 17" /><path d="M12 6h8M12 12h8M12 18h8" /></svg>);
}
function IconWand(props) {
  return (<svg {...iconBase} {...props}><path d="M4 20L18 6" /><path d="M15 3l1 2 2 1-2 1-1 2-1-2-2-1 2-1 1-2z" /><path d="M5 13l.6 1.4L7 15l-1.4.6L5 17l-.6-1.4L3 15l1.4-.6L5 13z" /></svg>);
}
function IconGlobe(props) {
  return (<svg {...iconBase} {...props}><circle cx="12" cy="12" r="9" /><path d="M3 12h18" /><path d="M12 3a14 14 0 010 18a14 14 0 010-18z" /></svg>);
}
function IconUpload(props) {
  return (<svg {...iconBase} {...props}><path d="M12 16V4" /><path d="M6 9l6-6 6 6" /><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3" /></svg>);
}
function IconEdit(props) {
  return (<svg {...iconBase} {...props}><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4L16.5 3.5z" /></svg>);
}

// Role/status pill — rounded, colored, sans, used across the admin area.
function Pill({ children, tone = "neutral" }) {
  const tones = {
    admin:     { bg: "#f3e8ff", fg: "#7c3aed" },
    user:      { bg: "#eef2ff", fg: "#4f46e5" },
    active:    { bg: "#dcfce7", fg: "#16a34a" },
    suspended: { bg: "#fee2e2", fg: "#dc2626" },
    neutral:   { bg: "#f1f5f9", fg: "#475569" },
    auto:      { bg: "#e0f2fe", fg: "#0369a1" },
    corrected: { bg: "#fef3c7", fg: "#b45309" },
    manual:    { bg: "#ede9fe", fg: "#6d28d9" },
    rated:     { bg: "#fef9c3", fg: "#a16207" },
  };
  const t = tones[tone] || tones.neutral;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "4px",
      padding: "2px 9px", borderRadius: "999px", fontSize: "10.5px", fontWeight: 700,
      background: t.bg, color: t.fg, whiteSpace: "nowrap", letterSpacing: ".01em",
    }}>{children}</span>
  );
}

const DOC_TYPE_COLOR = { case: "#e11d48", contract: "#2563eb", statute: "#059669" };
function IconScale(props) {
  return (<svg {...iconBase} {...props}><path d="M12 3v18M5 21h14" /><path d="M5 7l-3 6a3 3 0 006 0l-3-6zM19 7l-3 6a3 3 0 006 0l-3-6z" /><path d="M5 7h14M12 3L8 7h8l-4-4z" /></svg>);
}

// ─────────────────────────────────────────────────────────────────────────────
// TINY UI HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function Badge({ children, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px",
      background: color + "18", border: `1px solid ${color}40`,
      borderRadius: "3px", color, fontSize: "10px",
      fontFamily: "'Courier New',monospace", fontWeight: 700,
      letterSpacing: ".04em", whiteSpace: "nowrap",
    }}>{children}</span>
  );
}

function CitationChip({ page, label, onClick }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={() => onClick(page)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: "inline-flex", alignItems: "center", gap: "3px",
        padding: "2px 8px",
        background: hov ? "#bfdbfe" : "#dbeafe",
        border: "1px solid #3b82f6", borderRadius: "3px",
        color: "#1d4ed8", fontSize: "11px",
        fontFamily: "'Courier New',monospace",
        cursor: "pointer", fontWeight: 700, transition: "background .1s",
      }}
    >📄 p.{page}{label ? ` · ${label}` : ""}</button>
  );
}

function DotsLoader() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "6px 0" }}>
      <span style={{ fontSize: "10px", color: "#94a3b8", fontFamily: "'Courier New',monospace" }}>Analyzing…</span>
      <div style={{ display: "flex", gap: "3px" }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{
            width: "5px", height: "5px", borderRadius: "50%",
            background: "#3b82f6", display: "inline-block",
            animation: `dot-bounce 1.2s ${i * 0.2}s infinite ease-in-out`,
          }} />
        ))}
      </div>
    </div>
  );
}

function StatusDot({ color, text }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
      <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: color, flexShrink: 0 }} />
      <span style={{ fontSize: "10px", color: "#475569", fontFamily: "'Courier New',monospace", whiteSpace: "nowrap" }}>
        {text}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CHARTS — inline SVG, validated palette (dataviz skill: sequential blue, light mode)
// ─────────────────────────────────────────────────────────────────────────────
const VIZ = {
  series1:      "#2a78d6",
  series1Light: "#cde2fb",
  textPrimary:  "#0b0b0b",
  textSecondary:"#52514e",
  textMuted:    "#898781",
  gridline:     "#e1e0d9",
  baseline:     "#c3c2b7",
};

// Fixed-order categorical palette (never cycled) — for charts where color
// carries identity (different kinds of thing), not magnitude of the same kind.
const CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"];

// A rect with only the "data end" corner rounded (top for vertical bars, right for
// horizontal bars) and the baseline end square — plain SVG rect can't do per-corner
// radius, so a flat patch is overlaid on the baseline half to cancel that rounding.
function RoundEndRect({ x, y, w, h, radius = 4, roundEnd, fill }) {
  if (w <= 0 || h <= 0) return null;
  const r = Math.min(radius, roundEnd === "top" ? h : w);
  if (roundEnd === "top") {
    return (
      <>
        <rect x={x} y={y} width={w} height={h} rx={r} fill={fill} />
        {h > r && <rect x={x} y={y + h - r} width={w} height={r} fill={fill} />}
      </>
    );
  }
  return (
    <>
      <rect x={x} y={y} width={w} height={h} rx={r} fill={fill} />
      {w > r && <rect x={x} y={y} width={r} height={h} fill={fill} />}
    </>
  );
}

function niceMaxOf(values) {
  const raw = Math.max(...values, 0);
  if (raw <= 4) return 4;
  const step = Math.ceil(raw / 4);
  return step * 4;
}

// Trend over time — single series, sequential blue, area wash + 2px line,
// end-dot direct label, sparse axis labels (never one per point).
function TrendAreaChart({ data, width = 640, height = 170, color = VIZ.series1 }) {
  const padL = 30, padR = 14, padT = 18, padB = 22;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const n = data.length;
  const max = niceMaxOf(data.map(d => d.count));
  const x = i => padL + (n === 1 ? innerW / 2 : (innerW * i) / (n - 1));
  const y = v => padT + innerH - (max === 0 ? 0 : (innerH * v) / max);

  const linePath = data.map((d, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(d.count)}`).join(" ");
  const areaPath = `${linePath} L${x(n - 1)},${padT + innerH} L${x(0)},${padT + innerH} Z`;
  const last = data[n - 1];
  const labelStep = Math.max(1, Math.floor(n / 4));

  const fmtDate = iso => {
    const d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{ maxWidth: "100%", display: "block" }} preserveAspectRatio="none">
      {[0, 0.5, 1].map(f => (
        <g key={f}>
          <line x1={padL} y1={padT + innerH * f} x2={padL + innerW} y2={padT + innerH * f}
            stroke={VIZ.gridline} strokeWidth={1} />
          <text x={padL - 8} y={padT + innerH * f + 3} textAnchor="end" fontSize="10" fill={VIZ.textMuted}>
            {Math.round(max * (1 - f))}
          </text>
        </g>
      ))}
      <path d={areaPath} fill={color} opacity={0.1} />
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      {data.map((d, i) => (
        (i === 0 || i === n - 1 || i % labelStep === 0) && (
          <text key={d.date} x={x(i)} y={height - 4} textAnchor="middle" fontSize="10" fill={VIZ.textMuted}>
            {fmtDate(d.date)}
          </text>
        )
      ))}
      <circle cx={x(n - 1)} cy={y(last.count)} r={5} fill={color} stroke="#fff" strokeWidth={2} />
      <text x={x(n - 1)} y={y(last.count) - 12} textAnchor="end" fontSize="11.5" fontWeight={700} fill={VIZ.textPrimary}>
        {last.count}
      </text>
    </svg>
  );
}

// Compare magnitude across a handful of discrete categories — sequential blue,
// value labeled at each tip (fine for ≤6 bars; the "never label every point"
// rule targets dense line/scatter, not a short discrete bar comparison).
function BarChartVertical({ data, width = 300, height = 170, color = VIZ.series1 }) {
  const padL = 20, padR = 8, padT = 20, padB = 22;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const max = niceMaxOf(data.map(d => d.value));
  const n = data.length;
  const gap = 10;
  const barW = Math.min(24, (innerW - gap * (n - 1)) / n);
  const usedW = barW * n + gap * (n - 1);
  const startX = padL + (innerW - usedW) / 2;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{ maxWidth: "100%", display: "block" }} preserveAspectRatio="none">
      <line x1={padL} y1={padT + innerH} x2={padL + innerW} y2={padT + innerH} stroke={VIZ.baseline} strokeWidth={1} />
      {data.map((d, i) => {
        const h = max === 0 ? 0 : (d.value / max) * innerH;
        const bx = startX + i * (barW + gap);
        const by = padT + innerH - h;
        return (
          <g key={i}>
            <RoundEndRect x={bx} y={by} w={barW} h={h} roundEnd="top" fill={color} />
            <text x={bx + barW / 2} y={by - 5} textAnchor="middle" fontSize="10.5" fontWeight={600} fill={VIZ.textSecondary}>
              {d.value}
            </text>
            <text x={bx + barW / 2} y={padT + innerH + 15} textAnchor="middle" fontSize="10.5" fill={VIZ.textMuted}>
              {d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// Ranked comparison across named entities — horizontal so labels stay readable.
function BarChartHorizontal({ data, width = 300, color = VIZ.series1 }) {
  const rowH = 26, gap = 6, padL = 4, padR = 30, labelW = 88;
  const innerW = width - padL - padR - labelW;
  const max = niceMaxOf(data.map(d => d.value));
  const height = data.length * rowH + (data.length - 1) * gap;

  const safeHeight = Math.max(height, 1);
  return (
    <svg viewBox={`0 0 ${width} ${safeHeight}`} width="100%" height={safeHeight} style={{ maxWidth: "100%", display: "block" }} preserveAspectRatio="none">
      {data.map((d, i) => {
        const w = max === 0 ? 0 : (d.value / max) * innerW;
        const by = i * (rowH + gap);
        const barH = 16;
        const barY = by + (rowH - barH) / 2;
        const bx = padL + labelW;
        return (
          <g key={i}>
            <text x={padL + labelW - 8} y={by + rowH / 2 + 4} textAnchor="end" fontSize="11.5" fill={VIZ.textSecondary}
              style={{ overflow: "hidden" }}>
              {d.label.length > 12 ? d.label.slice(0, 11) + "…" : d.label}
            </text>
            <RoundEndRect x={bx} y={barY} w={w} h={barH} roundEnd="right" fill={color} />
            <text x={bx + w + 6} y={by + rowH / 2 + 4} fontSize="11" fontWeight={600} fill={VIZ.textPrimary}>
              {d.value}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// A single ratio against a limit — meter, not a two-slice pie.
function Meter({ value, max = 100, color = VIZ.series1, trackColor = VIZ.series1Light, label }) {
  const pct = max === 0 ? 0 : Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "7px" }}>
        <span style={{ fontSize: "11.5px", color: VIZ.textSecondary, fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: "13px", color: VIZ.textPrimary, fontWeight: 700 }}>{pct.toFixed(0)}%</span>
      </div>
      <div style={{ height: "10px", borderRadius: "6px", background: trackColor, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: "6px", transition: "width .4s" }} />
      </div>
    </div>
  );
}

// A trend folded INTO a stat tile, not a separate chart card — no axes, no
// gridlines, just the shape and the current value. This is how a real product
// dashboard (Stripe, Vercel) shows "this metric over time" without spending
// a whole card on a mostly-flat line.
function Sparkline({ data, color = VIZ.series1, width = 108, height = 30 }) {
  const n = data.length;
  if (!n) return null;
  const max = Math.max(...data.map(d => d.count), 1);
  const x = i => (n === 1 ? width / 2 : (width * i) / (n - 1));
  const y = v => height - (height - 3) * (v / max) - 1.5;
  const path = data.map((d, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(d.count).toFixed(1)}`).join(" ");
  const areaPath = `${path} L${x(n - 1).toFixed(1)},${height} L${x(0).toFixed(1)},${height} Z`;
  const last = data[n - 1];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <path d={areaPath} fill={color} opacity={0.12} />
      <path d={path} fill="none" stroke={color} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(n - 1)} cy={y(last.count)} r="2.5" fill={color} stroke="#fff" strokeWidth="1.2" />
    </svg>
  );
}

// When activity happens — weekday × hour grid, one hue, intensity by opacity.
// A genuinely different form from a bar/line: magnitude over TWO dimensions at once.
function ActivityHeatmap({ cells, color = VIZ.series1 }) {
  const max = Math.max(...cells.map(c => c.count), 1);
  const byKey = {};
  cells.forEach(c => { byKey[`${c.dow}-${c.hour}`] = c.count; });
  const hourTicks = [0, 6, 12, 18];
  const rowCols = "30px repeat(24, 1fr)"; // label rail + 24 fluid columns — fills whatever width the card has

  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "grid", gridTemplateColumns: rowCols, gap: "3px", marginBottom: "5px" }}>
        <div />
        {Array.from({ length: 24 }, (_, h) => (
          <div key={h} style={{ fontSize: "9px", color: VIZ.textMuted, textAlign: "center" }}>
            {hourTicks.includes(h) ? (h === 0 ? "12a" : h < 12 ? `${h}a` : h === 12 ? "12p" : `${h - 12}p`) : ""}
          </div>
        ))}
      </div>
      {WEEKDAY_LABELS.map((label, dow) => (
        <div key={label} style={{ display: "grid", gridTemplateColumns: rowCols, gap: "3px", marginBottom: "3px", alignItems: "center" }}>
          <div style={{ fontSize: "10px", color: VIZ.textMuted, textAlign: "right", paddingRight: "4px" }}>{label}</div>
          {Array.from({ length: 24 }, (_, hour) => {
            const count = byKey[`${dow}-${hour}`] || 0;
            const ratio = count / max;
            return (
              <div key={hour} title={`${label} ${hour}:00 — ${count} chat${count !== 1 ? "s" : ""}`} style={{
                width: "100%", aspectRatio: "1", borderRadius: "3px",
                background: count === 0 ? VIZ.gridline : color,
                opacity: count === 0 ? 0.5 : Math.max(0.16, ratio),
              }} />
            );
          })}
        </div>
      ))}
    </div>
  );
}

const WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

// Ranked list where each row is a different KIND of thing (not the same kind at
// different magnitudes) — categorical color, fixed order, never cycled.
function CategoricalBarList({ data, maxRows = 8 }) {
  const rows = data.slice(0, maxRows);
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  const max = Math.max(...rows.map(d => d.value), 1);
  return (
    <div>
      {rows.map((d, i) => {
        const color = CATEGORICAL[i % CATEGORICAL.length];
        const pct = (d.value / total) * 100;
        return (
          // Keyed by index — see DonutChart's identical note on why d.label
          // isn't safe as a key here.
          <div key={i} style={{ marginBottom: i === rows.length - 1 ? 0 : "10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "4px" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11.5px", color: VIZ.textSecondary, fontWeight: 600, minWidth: 0 }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "2px", background: color, flexShrink: 0 }} />
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.label}</span>
              </span>
              <span style={{ fontSize: "11px", color: VIZ.textPrimary, fontWeight: 700, flexShrink: 0, marginLeft: "8px" }}>
                {d.value} <span style={{ color: VIZ.textMuted, fontWeight: 500 }}>({pct.toFixed(0)}%)</span>
              </span>
            </div>
            <div style={{ height: "6px", borderRadius: "4px", background: "#f1f0ec", overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${(d.value / max) * 100}%`, background: color, borderRadius: "4px" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// A single ratio as a ring instead of a bar — same job as Meter, different
// mark, for when a page already has enough linear bars and needs visual variety.
function RadialProgress({ value, max = 100, color = VIZ.series1, trackColor = "#f1f0ec", size = 118, thickness = 13, label, sublabel }) {
  const pct = max === 0 ? 0 : Math.max(0, Math.min(100, (value / max) * 100));
  const r = (size - thickness) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - pct / 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
      <svg width={size} height={size} style={{ flexShrink: 0, transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={trackColor} strokeWidth={thickness} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={thickness}
          strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset .6s ease" }} />
      </svg>
      <div>
        <div style={{ fontSize: "27px", fontWeight: 700, color: VIZ.textPrimary }}>{pct.toFixed(0)}%</div>
        {label && <div style={{ fontSize: "12px", fontWeight: 600, color: VIZ.textSecondary, marginTop: "3px" }}>{label}</div>}
        {sublabel && <div style={{ fontSize: "11px", color: VIZ.textMuted, marginTop: "2px" }}>{sublabel}</div>}
      </div>
    </div>
  );
}

// A small ring with the raw total (not a percentage) in the center — the
// "Projects / Total 108" card pattern: icon chip top-left, ring top-right
// showing genuine recent-activity share, headline number and label below.
function MiniRing({ value, max, color, trackColor = "#eef0f3", size = 54, thickness = 6, centerText }) {
  const pct = max ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  const r = (size - thickness) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - pct / 100);
  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={trackColor} strokeWidth={thickness} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={thickness}
          strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" />
      </svg>
      {centerText && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "11px", fontWeight: 700, color: VIZ.textPrimary }}>
          {centerText}
        </div>
      )}
    </div>
  );
}

function RingStatCard({ icon: Icon, tint, label, sublabel, total, ringValue, ringMax }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #eef0f3", borderRadius: "16px", padding: "18px",
      boxShadow: CARD_SHADOW, display: "flex", flexDirection: "column", gap: "16px", minWidth: 0,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{
          width: "42px", height: "42px", borderRadius: "12px", background: tint, color: "#fff",
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
          boxShadow: `0 3px 10px ${tint}40`,
        }}><Icon width={20} height={20} /></div>
        <MiniRing value={ringValue} max={ringMax} color={tint} centerText={String(total)} />
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: "14.5px", fontWeight: 700, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</div>
        <div style={{ fontSize: "11px", color: "#9ca3af", marginTop: "2px" }}>{sublabel}</div>
      </div>
    </div>
  );
}

// Real part-to-whole with a legend — capped at a handful of categories (the
// dataviz all-pairs limit for a form like this), unlike the task-mode list
// which intentionally uses a bar for its 6+ categories instead.
function DonutChart({ data, size = 140, thickness = 20, layout = "row" }) {
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  const r = (size - thickness) / 2;
  const circumference = 2 * Math.PI * r;
  let cumulative = 0;
  const stacked = layout === "column";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: stacked ? "16px" : "22px",
      flexWrap: stacked ? "nowrap" : "wrap", flexDirection: stacked ? "column" : "row",
    }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)", flexShrink: 0 }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#f1f0ec" strokeWidth={thickness} />
        {data.map((d, i) => {
          const frac = d.value / total;
          const dash = circumference * frac;
          const gap = circumference - dash;
          const dashOffset = -cumulative * circumference;
          cumulative += frac;
          return (
            // Keyed by index, not d.label — two categories can legitimately
            // share the same display label (e.g. two "Other" buckets from
            // different groupings), and a duplicate key makes React silently
            // duplicate or drop a segment instead of just warning about it.
            <circle key={i} cx={size / 2} cy={size / 2} r={r} fill="none"
              stroke={CATEGORICAL[i % CATEGORICAL.length]} strokeWidth={thickness}
              strokeDasharray={`${dash} ${gap}`} strokeDashoffset={dashOffset} />
          );
        })}
      </svg>
      <div style={{
        display: "flex", flexDirection: stacked ? "row" : "column", flexWrap: stacked ? "wrap" : "nowrap",
        gap: stacked ? "14px" : "7px", minWidth: 0, justifyContent: stacked ? "center" : "flex-start",
      }}>
        {data.map((d, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "7px", fontSize: "11.5px", color: VIZ.textSecondary }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "2px", background: CATEGORICAL[i % CATEGORICAL.length], flexShrink: 0 }} />
            <span style={{ fontWeight: 700, color: VIZ.textPrimary }}>{d.value}</span>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Daily magnitude (bars — handles zero/near-zero days honestly, unlike a
// smoothed area that exaggerates flatness) plus a short moving-average trend
// line on the same scale. The dashboard's one "hero" chart.
function ComboBarLine({ data, width = 640, height = 200, barColor = VIZ.series1, lineColor = "#111827" }) {
  const padL = 32, padR = 14, padT = 18, padB = 24;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const n = data.length;
  const max = niceMaxOf(data.map(d => d.count));
  const barGap = 3;
  const barW = Math.max(4, innerW / n - barGap);
  const slotX = i => padL + (innerW * i) / n;
  const barX = i => slotX(i) + (innerW / n - barW) / 2;
  const lineX = i => slotX(i) + (innerW / n) / 2;
  const y = v => padT + innerH - (max === 0 ? 0 : (innerH * v) / max);

  const trend = data.map((d, i) => {
    const window = data.slice(Math.max(0, i - 1), i + 2);
    return window.reduce((s, w) => s + w.count, 0) / window.length;
  });
  const linePath = trend.map((v, i) => `${i === 0 ? "M" : "L"}${lineX(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  const fmtDate = iso => { const d = new Date(iso + "T00:00:00"); return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }); };
  const labelStep = Math.max(1, Math.floor(n / 5));
  const last = data[n - 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{ display: "block", maxWidth: "100%" }} preserveAspectRatio="none">
      {[0, 0.5, 1].map(f => (
        <g key={f}>
          <line x1={padL} y1={padT + innerH * f} x2={padL + innerW} y2={padT + innerH * f} stroke={VIZ.gridline} strokeWidth={1} />
          <text x={padL - 8} y={padT + innerH * f + 3} textAnchor="end" fontSize="10" fill={VIZ.textMuted}>{Math.round(max * (1 - f))}</text>
        </g>
      ))}
      {data.map((d, i) => (
        <RoundEndRect key={d.date} x={barX(i)} y={y(d.count)} w={barW} h={innerH - (y(d.count) - padT)} roundEnd="top" fill={barColor + "b3"} />
      ))}
      <path d={linePath} fill="none" stroke={lineColor} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={lineX(n - 1)} cy={y(trend[n - 1])} r={4} fill={lineColor} stroke="#fff" strokeWidth={1.5} />
      {data.map((d, i) => (
        (i === 0 || i === n - 1 || i % labelStep === 0) && (
          <text key={d.date} x={lineX(i)} y={height - 4} textAnchor="middle" fontSize="10" fill={VIZ.textMuted}>{fmtDate(d.date)}</text>
        )
      ))}
      <text x={lineX(n - 1)} y={y(last.count) - 10 < 12 ? 14 : y(last.count) - 10} textAnchor="middle" fontSize="11" fontWeight={700} fill={VIZ.textPrimary}>{last.count}</text>
    </svg>
  );
}

function ChartCard({ title, subtitle, action, children }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "18px",
      boxShadow: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: subtitle ? "1px" : "14px" }}>
        <div style={{ fontSize: "12.5px", fontWeight: 700, color: "#111827" }}>{title}</div>
        {action}
      </div>
      {subtitle && <div style={{ fontSize: "10.5px", color: "#94a3b8", marginBottom: "14px" }}>{subtitle}</div>}
      {children}
    </div>
  );
}

// Confirmation for destructive admin actions — replaces window.confirm with
// something that matches the rest of the interface.
function ConfirmDialog({ title, message, confirmLabel = "Confirm", danger = true, onConfirm, onCancel }) {
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 300, background: "rgba(15,17,20,.45)",
      display: "flex", alignItems: "center", justifyContent: "center",
    }} onClick={onCancel}>
      <div onClick={e => e.stopPropagation()} style={{
        background: "#fff", borderRadius: "14px", padding: "22px", width: "360px",
        boxShadow: "0 20px 40px rgba(0,0,0,.2)", fontFamily: UI_FONT,
      }}>
        <div style={{
          width: "38px", height: "38px", borderRadius: "10px",
          background: danger ? "#fee2e2" : "#eef2ff", color: danger ? "#dc2626" : "#4f46e5",
          display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "14px",
        }}><IconAlertTriangle width={19} height={19} /></div>
        <div style={{ fontSize: "14.5px", fontWeight: 700, color: "#111827", marginBottom: "6px" }}>{title}</div>
        <div style={{ fontSize: "12.5px", color: "#6b7280", lineHeight: "1.55", marginBottom: "20px" }}>{message}</div>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
          <button onClick={onCancel} style={{
            background: "#f8fafc", border: "1px solid #e2e8f0", color: "#374151",
            padding: "8px 14px", borderRadius: "8px", cursor: "pointer", fontSize: "12.5px", fontFamily: UI_FONT, fontWeight: 600,
          }}>Cancel</button>
          <button onClick={onConfirm} style={{
            background: danger ? "#dc2626" : "#2563eb", border: "none", color: "#fff",
            padding: "8px 14px", borderRadius: "8px", cursor: "pointer", fontSize: "12.5px", fontFamily: UI_FONT, fontWeight: 600,
          }}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF VIEWER — renders each page onto its own <canvas>
// ─────────────────────────────────────────────────────────────────────────────
function PdfCanvasPage({ pdfDoc, pageNum, highlighted }) {
  const canvasRef    = useRef(null);
  const textLayerRef = useRef(null);
  const renderingRef = useRef(false);

  useEffect(() => {
    if (!pdfDoc || !canvasRef.current || renderingRef.current) return;
    renderingRef.current = true;

    pdfDoc.getPage(pageNum).then(page => {
      const container = canvasRef.current?.parentElement;
      const containerWidth = container ? container.clientWidth - 2 : 640;
      const unscaledViewport = page.getViewport({ scale: 1 });
      const scale = containerWidth / unscaledViewport.width;
      const viewport = page.getViewport({ scale });

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      canvas.width  = viewport.width;
      canvas.height = viewport.height;
      // Make canvas fill container width via CSS
      canvas.style.width  = "100%";
      canvas.style.height = "auto";
      canvas.style.display = "block";

      page.render({ canvasContext: ctx, viewport }).promise.then(() => {
        renderingRef.current = false;
      });

      // Real, selectable text — invisible, laid directly on top of the
      // canvas glyphs at matching positions/size, purely so the browser's
      // native text selection (and our "Ask about selection" popover) has
      // something to select. The canvas itself is just pixels.
      if (textLayerRef.current) {
        textLayerRef.current.innerHTML = "";
        textLayerRef.current.style.setProperty("--scale-factor", String(viewport.scale));
        page.getTextContent().then(textContent => {
          if (!textLayerRef.current) return;
          pdfjsLib.renderTextLayer({
            textContentSource: textContent,
            container: textLayerRef.current,
            viewport,
          });
        }).catch(() => { /* text extraction failing shouldn't block viewing the page */ });
      }
    });
  }, [pdfDoc, pageNum]);

  return (
    <div
      id={`vpg-${pageNum}`}
      style={{
        position: "relative", width: "100%",
        background: "#fff",
        border: highlighted ? "2px solid #3b82f6" : "1px solid #d1d5db",
        boxShadow: highlighted
          ? "0 0 0 3px rgba(59,130,246,.2), 0 2px 8px rgba(0,0,0,.1)"
          : "0 1px 3px rgba(0,0,0,.06)",
        transition: "border-color .2s, box-shadow .2s",
        overflow: "hidden",
      }}
    >
      {/* page number */}
      <div style={{
        position: "absolute", top: 6, right: 8, zIndex: 3,
        fontSize: "9px", color: "#9ca3af",
        fontFamily: "'Courier New',monospace",
        background: "rgba(255,255,255,.9)", padding: "1px 5px",
        borderRadius: "2px", pointerEvents: "none",
      }}>
        {pageNum}
      </div>

      {/* highlighted badge */}
      {highlighted && (
        <div style={{
          position: "absolute", top: 6, left: 8, zIndex: 3,
          fontSize: "10px", fontWeight: 700, color: "#1d4ed8",
          background: "#dbeafe", border: "1px solid #93c5fd",
          padding: "2px 8px", borderRadius: "2px",
          fontFamily: "'Courier New',monospace",
          pointerEvents: "none",
        }}>▲ Referenced in answer</div>
      )}

      {/* yellow bottom bar on highlighted page */}
      {highlighted && (
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          height: "4px", background: "#fde047", zIndex: 3,
        }} />
      )}

      <canvas ref={canvasRef} />
      <div ref={textLayerRef} className="pdfTextLayer" style={{ position: "absolute", top: 0, left: 0, zIndex: 2 }} />
    </div>
  );
}

function PdfViewer({ doc, highlightPage }) {
  const [pdfDoc,    setPdfDoc]    = useState(null);
  const [pageCount, setPageCount] = useState(0);
  const [loading,   setLoading]   = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    setPdfDoc(null);
    setPageCount(0);
    if (!doc?.isPdf || !doc?.arrayBuffer) return;
    setLoading(true);
    pdfjsLib.getDocument({ data: doc.arrayBuffer.slice(0) }).promise
      .then(pdf => { setPdfDoc(pdf); setPageCount(pdf.numPages); setLoading(false); })
      .catch(() => setLoading(false));
  }, [doc]);

  useEffect(() => {
    if (!highlightPage) return;
    const el = document.getElementById(`vpg-${highlightPage}`);
    if (el) setTimeout(() => el.scrollIntoView({ behavior: "smooth", block: "center" }), 80);
  }, [highlightPage]);

  if (!doc) {
    return (
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        background: "#e9ecef", gap: "10px",
      }}>
        <div style={{ fontSize: "52px", opacity: .2 }}>📄</div>
        <div style={{ color: "#9ca3af", fontSize: "13px", fontFamily: "'Courier New',monospace" }}>
          No document open
        </div>
        <div style={{ color: "#cbd5e1", fontSize: "11px" }}>Upload a document to view it here</div>
      </div>
    );
  }

  // DOCX — rendered with docx-preview (pixel-perfect, no backend needed)
  if (!doc.isPdf && doc.isDocx) {
    return <DocxViewer doc={doc} highlightPage={highlightPage} />;
  }

  // TXT — clean text pager
  if (!doc.isPdf && !doc.isDocx) {
    const words    = (doc.text || "").split(/\s+/).filter(Boolean);
    const PER_PAGE = 350;
    const textPages = [];
    for (let i = 0; i < words.length; i += PER_PAGE) {
      textPages.push(words.slice(i, i + PER_PAGE).join(" "));
    }
    if (!textPages.length) textPages.push("(empty document)");

    return (
      <div ref={containerRef} style={{
        flex: 1, overflow: "auto", background: "#e9ecef",
        padding: "16px", display: "flex", flexDirection: "column",
        alignItems: "center", gap: "12px",
      }}>
        {textPages.map((pageText, i) => {
          const pg = i + 1;
          const hi = pg === highlightPage;
          return (
            <div key={pg} id={`vpg-${pg}`} style={{
              width: "100%", maxWidth: "680px", background: "#fff",
              border: hi ? "2px solid #3b82f6" : "1px solid #d1d5db",
              boxShadow: hi ? "0 0 0 3px rgba(59,130,246,.15)" : "0 1px 3px rgba(0,0,0,.06)",
              borderRadius: "2px", transition: "all .2s", overflow: "hidden",
            }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                padding: "7px 16px", background: hi ? "#eff6ff" : "#f8fafc",
                borderBottom: "1px solid #e5e7eb",
              }}>
                <span style={{
                  fontSize: "9px", color: "#9ca3af",
                  fontFamily: "'Courier New',monospace",
                  textTransform: "uppercase", letterSpacing: ".06em",
                  overflow: "hidden", textOverflow: "ellipsis",
                  whiteSpace: "nowrap", maxWidth: "70%",
                }}>{doc.name}</span>
                <span style={{ fontSize: "9px", color: "#9ca3af", fontFamily: "'Courier New',monospace" }}>
                  {pg} / {textPages.length}
                </span>
              </div>
              {hi && (
                <div style={{
                  background: "#dbeafe", borderBottom: "1px solid #bfdbfe",
                  padding: "5px 16px", fontSize: "10px", fontWeight: 700,
                  color: "#1d4ed8", fontFamily: "'Courier New',monospace",
                }}>▲ Referenced in answer</div>
              )}
              <div style={{
                padding: "24px 32px", fontSize: "12.5px", color: "#1e293b",
                lineHeight: "1.8", fontFamily: "Georgia,'Times New Roman',serif",
                background: hi ? "#fffbeb" : "#fff",
                whiteSpace: "pre-wrap",
                // pre-wrap alone only breaks at spaces — a long run with no
                // spaces (a URL, hash, or in this case mashed test text)
                // just overflows the card instead of wrapping. This breaks
                // it as a last resort, without affecting normal prose.
                overflowWrap: "anywhere",
              }}>{pageText}</div>
              {hi && <div style={{ height: "4px", background: "#fde047" }} />}
            </div>
          );
        })}
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", background: "#e9ecef" }}>
        <div style={{ fontSize: "11px", color: "#94a3b8", fontFamily: "'Courier New',monospace" }}>
          Loading {doc.name}…
        </div>
      </div>
    );
  }

  if (!pdfDoc) return null;

  return (
    <div ref={containerRef} style={{
      flex: 1, overflow: "auto", background: "#e9ecef",
      padding: "16px", display: "flex", flexDirection: "column",
      alignItems: "center", gap: "12px",
    }}>
      {Array.from({ length: pageCount }, (_, i) => i + 1).map(pg => (
        <div key={pg} style={{ width: "100%", maxWidth: "680px" }}>
          <PdfCanvasPage
            pdfDoc={pdfDoc}
            pageNum={pg}
            highlighted={pg === highlightPage}
          />
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FILE PARSERS
// ─────────────────────────────────────────────────────────────────────────────
async function parsePDF(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async e => {
      try {
        const arrayBuffer = e.target.result;
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer.slice(0) }).promise;
        let text = "";
        for (let i = 1; i <= pdf.numPages; i++) {
          const page    = await pdf.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map(it => it.str).join(" ") + "\n";
        }
        resolve({ text, pages: pdf.numPages, arrayBuffer, isPdf: true });
      } catch (err) { reject(err); }
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

async function parseDOCX(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async e => {
      const arrayBuffer = e.target.result;
      let plainText = "";

      try {
        // DOCX is a ZIP file. Extract word/document.xml and parse it cleanly.
        // We use the browser's built-in DecompressionStream to read the zip entries.
        // Render into a hidden div, then extract text — skipping style/script tags
        const hiddenDiv = document.createElement("div");
        hiddenDiv.style.cssText = "position:absolute;left:-9999px;top:-9999px;width:800px;pointer-events:none";
        document.body.appendChild(hiddenDiv);

        await renderAsync(
          arrayBuffer.slice(0),
          hiddenDiv,
          null,
          { className: "docx-text-extract", ignoreWidth: true, ignoreHeight: true, renderHeaders: false, renderFooters: false }
        );

        // Remove all style and script elements before extracting text
        hiddenDiv.querySelectorAll("style, script, link").forEach(el => el.remove());

        // Extract text only from actual content elements
        const contentEl = hiddenDiv.querySelector(".docx-text-extract") || hiddenDiv;
        plainText = (contentEl.textContent || contentEl.innerText || "")
          .replace(/\s+/g, " ")
          .trim()
          .slice(0, 50000);

        document.body.removeChild(hiddenDiv);
      } catch (_) {
        // If extraction fails, text stays empty — suggestions will use filename
      }

      const wordCount  = plainText.split(/\s+/).filter(Boolean).length;
      const approxPages = Math.max(1, Math.ceil(file.size / 3000));

      resolve({
        text:        plainText,
        arrayBuffer: arrayBuffer.slice(0),
        pages:       approxPages,
        wordCount,
        isPdf:       false,
        isDocx:      true,
      });
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

// ── Build HTML from raw text when mammoth finds no structure ──────────────────
async function parseTXT(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => {
      const text  = e.target.result;
      const words = text.split(/\s+/).filter(Boolean).length;
      resolve({ text, pages: Math.ceil(words / 350), isPdf: false });
    };
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

// ── DOCX viewer using docx-preview ───────────────────────────────────────────
function DocxViewer({ doc }) {
  const containerRef = useRef(null);
  const [rendered,  setRendered]  = useState(false);
  const [error,     setError]     = useState(null);

  useEffect(() => {
    if (!doc?.arrayBuffer || !containerRef.current) return;
    setRendered(false);
    setError(null);

    renderAsync(
      doc.arrayBuffer.slice(0),
      containerRef.current,
      null,
      {
        className:                   "docx-preview",
        inWrapper:                   true,
        ignoreWidth:                 true,
        ignoreHeight:                true,
        ignoreFonts:                 false,
        breakPages:                  true,
        ignoreLastRenderedPageBreak: false,
        useBase64URL:                true,
        renderChanges:               false,
        renderHeaders:               true,
        renderFooters:               true,
        renderFootnotes:             true,
        renderEndnotes:              true,
        renderComments:              false,
      }
    )
      .then(() => setRendered(true))
      .catch(err => { setError(err.message); setRendered(true); });
  }, [doc]);

  return (
    <div style={{
      flex: 1,
      overflow: "auto",
      minHeight: 0,
      background: "#e9ecef",
      padding: "16px",
      boxSizing: "border-box",
    }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>

        {/* toolbar */}
        <div style={{
          width: "100%", maxWidth: "780px",
          background: "#fff", border: "1px solid #d1d5db",
          borderBottom: "1px solid #e5e7eb",
          borderRadius: "2px 2px 0 0",
          padding: "7px 16px", display: "flex",
          justifyContent: "space-between", alignItems: "center",
        }}>
          <span style={{
            fontSize: "10px", color: "#9ca3af",
            fontFamily: "\'Courier New\',monospace",
            textTransform: "uppercase", letterSpacing: ".06em",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "80%",
          }}>{doc.name}</span>
          <span style={{ fontSize: "10px", color: "#6b7280", fontFamily: "\'Courier New\',monospace" }}>
            Word Document
          </span>
        </div>

        {/* loading */}
        {!rendered && !error && (
          <div style={{
            width: "100%", maxWidth: "780px", background: "#fff",
            border: "1px solid #d1d5db", borderTop: "none",
            padding: "60px", display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <span style={{ fontSize: "12px", color: "#94a3b8", fontFamily: "\'Courier New\',monospace" }}>
              Rendering document…
            </span>
          </div>
        )}

        {/* error */}
        {error && (
          <div style={{
            width: "100%", maxWidth: "780px", background: "#fff",
            border: "1px solid #fca5a5", borderTop: "none",
            padding: "20px", color: "#991b1b", fontSize: "12px",
          }}>
            Could not render: {error}
          </div>
        )}

        {/* docx-preview mount point — height grows with content, no overflow:hidden */}
        <div
          ref={containerRef}
          style={{
            width: "100%", maxWidth: "780px",
            background: "#fff",
            border: "1px solid #d1d5db",
            borderTop: "none",
            borderRadius: "0 0 2px 2px",
            visibility: rendered ? "visible" : "hidden",
            overflowX: "auto",
          }}
        />

      </div>

      <style>{`
        .docx-preview-wrapper { padding: 0 !important; background: transparent !important; }
        .docx-preview section.docx {
          width: 100% !important;
          max-width: 100% !important;
          padding: 48px 64px !important;
          box-shadow: none !important;
          margin: 0 !important;
          border-bottom: 1px solid #e5e7eb !important;
          box-sizing: border-box !important;
          background: #fff !important;
          overflow-x: auto !important;
        }
        .docx-preview section.docx:last-child { border-bottom: none !important; }
        .docx-preview { font-family: Arial, sans-serif; width: 100% !important; }
        .docx-preview img { max-width: 100%; height: auto; }
        /* Word tables carry fixed pixel/point widths from the original document —
           without this they overflow the panel and get clipped on the right edge. */
        .docx-preview table {
          table-layout: fixed !important;
          width: 100% !important;
          max-width: 100% !important;
          box-sizing: border-box !important;
        }
        .docx-preview table td, .docx-preview table th {
          word-break: break-word !important;
          overflow-wrap: break-word !important;
          box-sizing: border-box !important;
        }
        .docx-preview pre, .docx-preview code {
          white-space: pre-wrap !important;
          word-break: break-word !important;
        }
      `}</style>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPOSER — ChatGPT-style pill input bar with inline attach button
// ─────────────────────────────────────────────────────────────────────────────
function Composer({ input, setInput, onSend, onAttach, uploading, disabled, placeholder, autoFocus }) {
  const taRef = useRef(null);

  useEffect(() => {
    if (!taRef.current) return;
    taRef.current.style.height = "auto";
    taRef.current.style.height = Math.min(taRef.current.scrollHeight, 200) + "px";
  }, [input]);

  // The plain `autoFocus` HTML attribute only fires once, at mount — it
  // won't refocus a textarea that's already on the page. This component
  // never unmounts as `autoFocus` toggles (e.g. confirming a text
  // selection to ask about), so re-focus explicitly whenever it flips true.
  useEffect(() => {
    if (autoFocus) taRef.current?.focus();
  }, [autoFocus]);

  return (
    <div style={{
      display: "flex", alignItems: "flex-end", gap: "4px",
      background: "#fff", border: "1px solid #d9d9e3", borderRadius: "26px",
      padding: "7px 8px 7px 8px", boxShadow: "0 2px 12px rgba(0,0,0,.05)",
    }}>
      <button
        onClick={onAttach}
        disabled={uploading}
        title="Attach PDF, DOCX or TXT"
        style={{
          background: "none", border: "none", flexShrink: 0,
          cursor: uploading ? "default" : "pointer", color: "#6b7280",
          width: "34px", height: "34px", borderRadius: "50%",
          display: "flex", alignItems: "center", justifyContent: "center",
          transition: "background .12s",
        }}
        onMouseEnter={e => e.currentTarget.style.background = "#f3f4f6"}
        onMouseLeave={e => e.currentTarget.style.background = "transparent"}
      ><IconPaperclip /></button>

      <textarea
        ref={taRef}
        autoFocus={autoFocus}
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => {
          if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
        }}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        style={{
          flex: 1, border: "none", outline: "none", resize: "none",
          fontSize: "14.5px", lineHeight: "1.5", fontFamily: UI_FONT,
          color: "#0d0d0d", background: "transparent",
          padding: "7px 2px", maxHeight: "200px", overflowY: "auto",
        }}
      />

      <button
        onClick={onSend}
        disabled={disabled || !input.trim()}
        style={{
          background: (disabled || !input.trim()) ? "#e5e7eb" : "#0d0d0d",
          border: "none", color: "#fff", width: "34px", height: "34px",
          borderRadius: "50%", flexShrink: 0,
          cursor: (disabled || !input.trim()) ? "default" : "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          transition: "background .15s",
        }}
      ><IconArrowUp width={16} height={16} /></button>
    </div>
  );
}

const ALL_DOCS = "__all_docs__"; // sentinel activeDocId meaning "combined across every uploaded document"

// ── Small shared utilities for the Knowledge Base & Chat History views ──────
function timeAgo(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const sec = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day}d ago`;
  const mo = Math.floor(day / 30);
  if (mo < 12) return `${mo}mo ago`;
  return `${Math.floor(mo / 12)}y ago`;
}

function clampStyle(lines) {
  return { display: "-webkit-box", WebkitLineClamp: lines, WebkitBoxOrient: "vertical", overflow: "hidden" };
}

function downloadCsv(filename, headers, rows) {
  const esc = v => `"${String(v ?? "").replace(/"/g, '""').replace(/\r?\n/g, " ")}"`;
  const csv = [headers, ...rows].map(row => row.map(esc).join(",")).join("\r\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  a.click(); URL.revokeObjectURL(url);
}

function downloadJsonl(filename, objects) {
  const jsonl = objects.map(o => JSON.stringify(o)).join("\n");
  const blob = new Blob([jsonl], { type: "application/jsonl;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  a.click(); URL.revokeObjectURL(url);
}

// Heuristic: does this entry carry a human's judgment (a manual entry, an
// imported batch, or a correction someone actually made) rather than a raw,
// never-reviewed model guess? Used to let training exports include only
// higher-trust data.
function isKbVerified(entry) {
  const by = (entry.corrected_by || "").toLowerCase();
  return !by.includes("auto");
}

function formatDayHeader(dateKey) {
  if (!dateKey) return "Unknown date";
  const d = new Date(dateKey + "T00:00:00");
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1);
  const dOnly = new Date(d); dOnly.setHours(0, 0, 0, 0);
  if (dOnly.getTime() === today.getTime()) return "Today";
  if (dOnly.getTime() === yesterday.getTime()) return "Yesterday";
  return d.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });
}

function groupByDay(items, tsKey = "timestamp") {
  const groups = [];
  let currentKey = null, currentGroup = null;
  for (const item of items) {
    const key = (item[tsKey] || "").slice(0, 10);
    if (key !== currentKey) {
      currentKey = key;
      currentGroup = { date: key, items: [] };
      groups.push(currentGroup);
    }
    currentGroup.items.push(item);
  }
  return groups;
}

// Pure CourtListener lookup — shared by the Case Library search box and the
// Related Precedents tool (which calls it once per extracted legal issue).
async function courtListenerQuery(query, pageSize = 3) {
  try {
    const res = await fetch(
      `https://www.courtlistener.com/api/rest/v4/search/?q=${encodeURIComponent(query)}&type=o&format=json&page_size=${pageSize}`,
      { headers: { "Accept": "application/json" } }
    );
    if (!res.ok) throw new Error();
    const data = await res.json();
    return (data.results || []).map(r => ({
      name:    r.caseName || r.case_name || "Unknown Case",
      court:   (r.court || r.court_id || "").replace(/_/g, " "),
      date:    (r.dateFiled || r.date_filed || "").slice(0, 4),
      url:     r.absolute_url ? `https://www.courtlistener.com${r.absolute_url}` : null,
      snippet: r.snippet || "",
    }));
  } catch {
    return [];
  }
}

// Pull the first well-formed JSON value of the expected shape out of an LLM
// response — local models don't reliably emit JSON-only even when asked, so
// this tolerates a stray sentence before/after the actual payload.
function extractJson(text, kind) {
  const re = kind === "array" ? /\[[\s\S]*\]/ : /\{[\s\S]*\}/;
  const m = (text || "").match(re);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

const SEVERITY_TONE = { High: "suspended", Medium: "corrected", Low: "active" };
const CARD_SHADOW = "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)";
const toolCardStyle = { background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "16px", boxShadow: CARD_SHADOW };
const rerunBtnStyle = {
  display: "flex", alignItems: "center", gap: "6px", background: "#fff", border: "1px solid #e2e8f0",
  borderRadius: "9px", padding: "7px 13px", fontSize: "12px", fontWeight: 600, color: "#374151",
  cursor: "pointer", fontFamily: UI_FONT, flexShrink: 0, whiteSpace: "nowrap",
};

// ─────────────────────────────────────────────────────────────────────────────
// TASK-MODE TOOL PANELS — each mode gets a purpose-built view instead of a
// chat prompt: structured input where the tool needs it, structured output
// rendered as data (tables, badges, sections), not another chat bubble.
// ─────────────────────────────────────────────────────────────────────────────
function ToolHeader({ icon: Icon, color, title, subtitle, action }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "18px", gap: "12px", flexWrap: "wrap" }}>
      <div style={{ display: "flex", gap: "12px" }}>
        <div style={{
          width: "38px", height: "38px", borderRadius: "11px", background: color + "18", color,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}><Icon width={19} height={19} /></div>
        <div>
          <div style={{ fontSize: "16.5px", fontWeight: 700, color: "#111827" }}>{title}</div>
          <div style={{ fontSize: "12px", color: "#6b7280", marginTop: "2px", maxWidth: "440px", lineHeight: "1.5" }}>{subtitle}</div>
        </div>
      </div>
      {action}
    </div>
  );
}

function EmptyState({ text }) {
  return (
    <div style={{ padding: "32px 20px", textAlign: "center", background: "#fff", border: "1px dashed #e2e8f0", borderRadius: "14px", fontSize: "12.5px", color: "#9ca3af" }}>{text}</div>
  );
}

function LoadingCard({ text }) {
  return (
    <div style={{ ...toolCardStyle, display: "flex", alignItems: "center", gap: "10px", color: "#6b7280", fontSize: "12.5px" }}>
      <div style={{ display: "flex", gap: "3px" }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{ width: "5px", height: "5px", borderRadius: "50%", background: "#9ca3af", display: "inline-block", animation: `dot-bounce 1.2s ${i * 0.2}s infinite ease-in-out` }} />
        ))}
      </div>
      {text}
    </div>
  );
}

function IracSection({ label, text, color }) {
  return (
    <div>
      <div style={{ fontSize: "10px", fontWeight: 700, color, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: "4px" }}>{label}</div>
      <div style={{ fontSize: "12.5px", color: "#374151", lineHeight: "1.65" }}>{text || "—"}</div>
    </div>
  );
}

function ArgumentToolView({ argumentEntries, argumentClaim, setArgumentClaim, onRunArgument, toolLoading }) {
  return (
    <div>
      <ToolHeader icon={IconScale} color="#7c3aed" title="Argument Generator"
        subtitle="Build a structured legal argument in IRAC format from any claim or position." />

      <div style={{ ...toolCardStyle, marginBottom: "18px" }}>
        <label style={{ fontSize: "11.5px", fontWeight: 600, color: "#374151", marginBottom: "7px", display: "block" }}>
          What claim or position do you want to argue?
        </label>
        <textarea
          value={argumentClaim}
          onChange={e => setArgumentClaim(e.target.value)}
          placeholder='e.g. "The termination clause is unenforceable due to lack of notice"'
          rows={2}
          style={{ width: "100%", border: "1px solid #d1d5db", borderRadius: "10px", padding: "10px 12px", fontSize: "13px", fontFamily: UI_FONT, resize: "vertical", outline: "none", boxSizing: "border-box" }}
        />
        <button
          onClick={() => onRunArgument(argumentClaim)}
          disabled={!argumentClaim.trim() || toolLoading}
          style={{
            marginTop: "10px", display: "flex", alignItems: "center", gap: "7px",
            background: (!argumentClaim.trim() || toolLoading) ? "#e5e7eb" : "linear-gradient(180deg,#7c3aed,#6d28d9)",
            color: "#fff", border: "none", borderRadius: "9px", padding: "9px 16px", fontSize: "12.5px", fontWeight: 700,
            cursor: (!argumentClaim.trim() || toolLoading) ? "default" : "pointer", fontFamily: UI_FONT,
          }}
        ><IconWand width={14} height={14} />{toolLoading ? "Generating…" : "Generate Argument"}</button>
      </div>

      {toolLoading && argumentEntries.length === 0 && <LoadingCard text="Building the argument…" />}
      {argumentEntries.length === 0 && !toolLoading && (
        <EmptyState text="No arguments generated yet. Enter a claim above to build your first IRAC brief." />
      )}

      {argumentEntries.map(entry => (
        <div key={entry.id} style={{ ...toolCardStyle, marginBottom: "12px" }}>
          <div style={{ fontSize: "10.5px", color: "#9ca3af", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".04em", marginBottom: "3px" }}>Claim</div>
          <div style={{ fontSize: "13.5px", fontWeight: 700, color: "#111827", marginBottom: "16px", lineHeight: "1.4" }}>{entry.claim}</div>
          {entry.parsed ? (
            <div style={{ display: "grid", gap: "14px" }}>
              <IracSection label="Issue" text={entry.issue} color="#7c3aed" />
              <IracSection label="Rule" text={entry.rule} color="#2563eb" />
              <IracSection label="Application" text={entry.application} color="#059669" />
              <IracSection label="Conclusion" text={entry.conclusion} color="#111827" />
              {entry.weaknesses?.length > 0 && (
                <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: "10px", padding: "12px 14px" }}>
                  <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#c2410c", textTransform: "uppercase", letterSpacing: ".04em", marginBottom: "7px" }}>
                    ⚠ Weaknesses opposing counsel might exploit
                  </div>
                  <ul style={{ margin: 0, paddingLeft: "18px" }}>
                    {entry.weaknesses.map((w, i) => <li key={i} style={{ fontSize: "12.5px", color: "#7c2d12", lineHeight: "1.7" }}>{w}</li>)}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: "12.5px", color: "#6b7280", whiteSpace: "pre-wrap", lineHeight: "1.6" }}>{entry.raw}</div>
          )}
        </div>
      ))}
    </div>
  );
}

function RiskToolView({ riskResult, onRunRisk, toolLoading }) {
  const items = riskResult?.items;
  const counts = { High: 0, Medium: 0, Low: 0 };
  (items || []).forEach(r => { if (counts[r.severity] !== undefined) counts[r.severity]++; });

  return (
    <div>
      <ToolHeader icon={IconAlertTriangle} color="#dc2626" title="Risk Analysis"
        subtitle="Scans the document for ambiguous language, missing clauses, unfavorable terms, and jurisdiction risk."
        action={
          <button onClick={onRunRisk} disabled={toolLoading} style={rerunBtnStyle}>
            <IconRefresh width={13} height={13} />{riskResult ? "Re-scan" : "Scan Document"}
          </button>
        } />

      {toolLoading && !riskResult && <LoadingCard text="Scanning the document for risk factors…" />}

      {items && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px", marginBottom: "16px" }}>
            {["High", "Medium", "Low"].map(sev => (
              <div key={sev} style={{ ...toolCardStyle, padding: "14px 16px" }}>
                <div style={{ fontSize: "10px", color: "#94a3b8", textTransform: "uppercase", fontWeight: 700, marginBottom: "4px" }}>{sev} severity</div>
                <div style={{ fontSize: "22px", fontWeight: 700, color: sev === "High" ? "#dc2626" : sev === "Medium" ? "#d97706" : "#16a34a" }}>{counts[sev]}</div>
              </div>
            ))}
          </div>

          {items.length === 0 ? (
            <EmptyState text="No notable risks found in this document." />
          ) : items.map((r, i) => (
            <div key={i} style={{ ...toolCardStyle, marginBottom: "10px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "7px", gap: "10px" }}>
                <div style={{ fontSize: "13px", fontWeight: 700, color: "#111827" }}>{r.title}</div>
                <Pill tone={SEVERITY_TONE[r.severity] || "neutral"}>{r.severity}</Pill>
              </div>
              <div style={{ fontSize: "12.5px", color: "#4b5563", lineHeight: "1.6", marginBottom: "9px" }}>{r.description}</div>
              <div style={{ background: "#f8fafc", borderRadius: "8px", padding: "9px 11px", fontSize: "11.5px", color: "#374151", lineHeight: "1.5" }}>
                <strong>Recommendation:</strong> {r.recommendation}
              </div>
              {r.location && <div style={{ fontSize: "10.5px", color: "#9ca3af", marginTop: "7px" }}>📍 {r.location}</div>}
            </div>
          ))}
        </>
      )}

      {riskResult && !items && riskResult.raw && (
        <div style={toolCardStyle}><div style={{ fontSize: "12.5px", color: "#6b7280", whiteSpace: "pre-wrap" }}>{riskResult.raw}</div></div>
      )}
      {!riskResult && !toolLoading && <EmptyState text="Click Scan Document to run a full risk analysis." />}
    </div>
  );
}

function ClauseToolView({ clauseEntries, customClauseInput, setCustomClauseInput, onRunClause, toolLoading }) {
  return (
    <div>
      <ToolHeader icon={IconTag} color="#0891b2" title="Clause Extractor"
        subtitle="Pick a clause type to pull the exact text, a plain-English explanation, and a risk rating." />

      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "10px" }}>
        {CLAUSE_TYPES.map(ct => (
          <button key={ct} onClick={() => onRunClause(ct)} disabled={toolLoading} style={{
            background: "#fff", border: "1px solid #cffafe", color: "#0891b2", borderRadius: "999px",
            padding: "6px 13px", fontSize: "12px", fontWeight: 600, cursor: toolLoading ? "default" : "pointer", fontFamily: UI_FONT,
          }}>{ct}</button>
        ))}
      </div>
      <div style={{ display: "flex", gap: "8px", marginBottom: "18px" }}>
        <input
          value={customClauseInput}
          onChange={e => setCustomClauseInput(e.target.value)}
          placeholder="Or type a custom clause type…"
          onKeyDown={e => { if (e.key === "Enter" && customClauseInput.trim()) onRunClause(customClauseInput); }}
          style={{ flex: 1, border: "1px solid #d1d5db", borderRadius: "10px", padding: "9px 12px", fontSize: "13px", fontFamily: UI_FONT, outline: "none" }}
        />
        <button
          onClick={() => onRunClause(customClauseInput)}
          disabled={!customClauseInput.trim() || toolLoading}
          style={{
            background: (!customClauseInput.trim() || toolLoading) ? "#e5e7eb" : "#0891b2", color: "#fff",
            border: "none", borderRadius: "10px", padding: "0 18px", fontSize: "12.5px", fontWeight: 700,
            cursor: (!customClauseInput.trim() || toolLoading) ? "default" : "pointer", fontFamily: UI_FONT,
          }}
        >Extract</button>
      </div>

      {toolLoading && <LoadingCard text="Extracting clause…" />}
      {clauseEntries.length === 0 && !toolLoading && (
        <EmptyState text="Click a clause type above to build this document's clause library." />
      )}

      {clauseEntries.map(entry => (
        <div key={entry.id} style={{ ...toolCardStyle, marginBottom: "10px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "9px" }}>
            <div style={{ fontSize: "13px", fontWeight: 700, color: "#111827" }}>{entry.clauseType}</div>
            {entry.parsed && (
              entry.found
                ? <Pill tone={SEVERITY_TONE[entry.risk_level] || "neutral"}>{entry.risk_level} risk</Pill>
                : <Pill tone="neutral">Not found</Pill>
            )}
          </div>
          {entry.parsed ? (
            <>
              {entry.found && entry.exact_text && (
                <div style={{ background: "#f8fafc", borderLeft: "3px solid #0891b2", borderRadius: "6px", padding: "10px 12px", fontSize: "12px", color: "#374151", fontStyle: "italic", marginBottom: "9px", lineHeight: "1.6" }}>
                  "{entry.exact_text}"
                </div>
              )}
              <div style={{ fontSize: "12.5px", color: "#4b5563", lineHeight: "1.6" }}>{entry.explanation}</div>
            </>
          ) : <div style={{ fontSize: "12.5px", color: "#6b7280", whiteSpace: "pre-wrap" }}>{entry.raw}</div>}
        </div>
      ))}
    </div>
  );
}

function SummarySection({ title, items, icon: Icon }) {
  return (
    <div style={toolCardStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
        <Icon width={14} height={14} color="#059669" />
        <div style={{ fontSize: "11.5px", fontWeight: 700, color: "#111827", textTransform: "uppercase", letterSpacing: ".04em" }}>{title}</div>
      </div>
      {(!items || items.length === 0) ? (
        <div style={{ fontSize: "12px", color: "#9ca3af" }}>None identified.</div>
      ) : (
        <ul style={{ margin: 0, paddingLeft: "18px" }}>
          {items.map((it, i) => <li key={i} style={{ fontSize: "12.5px", color: "#374151", lineHeight: "1.75" }}>{it}</li>)}
        </ul>
      )}
    </div>
  );
}

function SummaryToolView({ summaryResult, summaryLength, setSummaryLength, onRunSummary, toolLoading }) {
  return (
    <div>
      <ToolHeader icon={IconListChecks} color="#059669" title="Document Brief"
        subtitle="A structured executive summary — parties, dates, obligations, and notable clauses."
        action={
          <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
            <select value={summaryLength} onChange={e => setSummaryLength(e.target.value)} style={{
              border: "1px solid #d1d5db", borderRadius: "8px", padding: "7px 9px", fontSize: "11.5px", fontFamily: UI_FONT, color: "#374151",
            }}>
              <option value="brief">Brief</option>
              <option value="standard">Standard</option>
              <option value="detailed">Detailed</option>
            </select>
            <button onClick={() => onRunSummary(summaryLength)} disabled={toolLoading} style={rerunBtnStyle}>
              <IconRefresh width={13} height={13} />{summaryResult ? "Regenerate" : "Generate"}
            </button>
          </div>
        } />

      {toolLoading && !summaryResult && <LoadingCard text="Reading the document…" />}
      {summaryResult?.parsed && (
        <div style={{ display: "grid", gap: "12px" }}>
          <SummarySection title="Parties Involved" items={summaryResult.parties} icon={IconUser} />
          <SummarySection title="Key Dates" items={(summaryResult.key_dates || []).map(d => `${d.date} — ${d.description}`)} icon={IconLayers} />
          <SummarySection title="Main Obligations" items={summaryResult.obligations} icon={IconShield} />
          <SummarySection title="Notable Clauses & Findings" items={summaryResult.notable_clauses} icon={IconTag} />
        </div>
      )}
      {summaryResult && !summaryResult.parsed && (
        <div style={toolCardStyle}><div style={{ fontSize: "12.5px", color: "#6b7280", whiteSpace: "pre-wrap" }}>{summaryResult.raw}</div></div>
      )}
      {!summaryResult && !toolLoading && <EmptyState text="Click Generate to build a structured brief of this document." />}
    </div>
  );
}

function PrecedentsToolView({ precedentsResult, onRunPrecedents, toolLoading }) {
  return (
    <div>
      <ToolHeader icon={IconBook} color="#7c3aed" title="Related Precedents"
        subtitle="AI extracts this document's key legal issues, then searches CourtListener for matching case law on each."
        action={
          <button onClick={onRunPrecedents} disabled={toolLoading} style={rerunBtnStyle}>
            <IconRefresh width={13} height={13} />{precedentsResult ? "Search again" : "Find Precedents"}
          </button>
        } />

      {toolLoading && !precedentsResult && <LoadingCard text="Identifying legal issues and searching case law…" />}
      {precedentsResult && precedentsResult.issues.length === 0 && (
        <EmptyState text="No clear legal issues could be identified for precedent search." />
      )}
      {precedentsResult?.issues.map((group, i) => (
        <div key={i} style={{ marginBottom: "18px" }}>
          <div style={{ fontSize: "12.5px", fontWeight: 700, color: "#111827", marginBottom: "9px", display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ width: "20px", height: "20px", borderRadius: "6px", background: "#f3e8ff", color: "#7c3aed", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "10px", fontWeight: 700, flexShrink: 0 }}>{i + 1}</span>
            {group.issue}
          </div>
          {group.cases.length === 0 ? (
            <div style={{ fontSize: "12px", color: "#9ca3af", paddingLeft: "28px" }}>No matching cases found on CourtListener.</div>
          ) : group.cases.map((c, ci) => (
            <div key={ci} onClick={() => c.url && window.open(c.url, "_blank")} style={{
              marginLeft: "28px", marginBottom: "6px", padding: "10px 12px", background: "#fff",
              border: "1px solid #eef0f3", borderRadius: "10px", cursor: c.url ? "pointer" : "default", boxShadow: CARD_SHADOW,
            }}>
              <div style={{ fontSize: "12.5px", fontWeight: 600, color: "#1d4ed8" }}>{c.name}</div>
              <div style={{ fontSize: "10.5px", color: "#9ca3af" }}>
                {c.court}{c.date ? ` · ${c.date}` : ""}{c.url && <span style={{ color: "#3b82f6", marginLeft: "6px" }}>↗</span>}
              </div>
            </div>
          ))}
        </div>
      ))}
      {!precedentsResult && !toolLoading && <EmptyState text="Click Find Precedents to identify key issues and search case law." />}
    </div>
  );
}

function ToolPanel(props) {
  const { task, hasDocs, toolError } = props;
  if (!hasDocs) return null;
  return (
    <div style={{ flex: 1, overflow: "auto" }}>
      <div style={{ maxWidth: "820px", margin: "0 auto", padding: "24px 20px 40px" }}>
        {toolError && (
          <div style={{ background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: "10px", padding: "10px 14px", color: "#991b1b", fontSize: "12px", marginBottom: "16px" }}>{toolError}</div>
        )}
        {task === "argument" && <ArgumentToolView {...props} />}
        {task === "risk" && <RiskToolView {...props} />}
        {task === "clause" && <ClauseToolView {...props} />}
        {task === "summarize" && <SummaryToolView {...props} />}
        {task === "precedents" && <PrecedentsToolView {...props} />}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function GridApp() {
  const [docs,        setDocs]        = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);
  const [activeTask,  setActiveTask]  = useState("research");
  const [messages,    setMessages]    = useState([]);
  const [input,       setInput]       = useState("");
  const [typing,      setTyping]      = useState(false);
  const [hlPage,      setHlPage]      = useState(null);
  const [uploading,   setUploading]   = useState(false);
  const [uploadErr,   setUploadErr]   = useState("");
  const [isDragging,  setIsDragging]  = useState(false);
  const [mainView,    setMainView]    = useState("chat"); // chat | precedents | kb
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sourceOpen,  setSourceOpen]  = useState(true);
  const [kbCount,     setKbCount]     = useState(0);
  const [suggestions, setSuggestions] = useState([]);
  const [precedents,  setPrecedents]  = useState([]);
  const [precLoading, setPrecLoading] = useState(false);

  // ── task-mode tools — each keyed by the active doc/scope id, so results
  // persist per document when you switch away and back ────────────────────
  const [toolLoading,     setToolLoading]     = useState(false);
  const [toolError,       setToolError]       = useState("");
  const [argumentByDoc,   setArgumentByDoc]   = useState({});
  const [argumentClaim,   setArgumentClaim]   = useState("");
  const [riskByDoc,       setRiskByDoc]       = useState({});
  const [clauseByDoc,     setClauseByDoc]     = useState({});
  const [customClauseInput, setCustomClauseInput] = useState("");
  const [summaryByDoc,    setSummaryByDoc]    = useState({});
  const [summaryLength,   setSummaryLength]   = useState("standard"); // brief | standard | detailed
  const [precedentsToolByDoc, setPrecedentsToolByDoc] = useState({});
  const [kbEntries,   setKbEntries]   = useState([]);
  const [kbLoading,   setKbLoading]   = useState(false);
  const [kbSearch,    setKbSearch]    = useState("");
  const [kbSort,      setKbSort]      = useState("usage"); // usage | recent | alpha
  const [expandedKb,  setExpandedKb]  = useState(() => new Set());
  const [kbCategory,   setKbCategory]   = useState("all"); // all | document | external_url | external_import | manual
  const [kbTool,        setKbTool]        = useState(null); // null | "url" | "manual" | "import"
  const [kbUrlInput,    setKbUrlInput]    = useState("");
  const [kbUrlBusy,     setKbUrlBusy]     = useState(false);
  const [kbUrlMsg,      setKbUrlMsg]      = useState(null); // { ok: bool, text }
  const [kbManualQ,     setKbManualQ]     = useState("");
  const [kbManualA,     setKbManualA]     = useState("");
  const [kbManualSrc,   setKbManualSrc]   = useState("");
  const [kbManualBusy,  setKbManualBusy]  = useState(false);
  const [kbManualMsg,   setKbManualMsg]   = useState(null);
  const [kbImportBusy,  setKbImportBusy]  = useState(false);
  const [kbImportMsg,   setKbImportMsg]   = useState(null);
  const [kbVerifiedOnly, setKbVerifiedOnly] = useState(false);
  const kbImportFileRef = useRef(null);

  // ── auth / admin ─────────────────────────────────────────────────────────
  const [authToken,    setAuthToken]    = useState(() => localStorage.getItem("authToken") || null);
  const [currentUser,  setCurrentUser]  = useState(null); // { username, email, is_admin }
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false); // header account dropdown
  const userMenuRef = useRef(null);
  const [notification, setNotification] = useState("");
  const [adminTab,      setAdminTab]      = useState("overview"); // overview | users | history | insights
  const [adminInsights, setAdminInsights] = useState(null);
  const [adminInsightsLoading, setAdminInsightsLoading] = useState(false);
  const [insightsAnalyzeBusy, setInsightsAnalyzeBusy] = useState(false);
  const [expandedInsightDocs, setExpandedInsightDocs] = useState(() => new Set());
  const [adminOverview, setAdminOverview] = useState(null);
  const [adminAnalytics, setAdminAnalytics] = useState(null);
  const [adminAuditLog, setAdminAuditLog] = useState([]);
  const [adminSystemStatus, setAdminSystemStatus] = useState(null);
  const [adminChats,    setAdminChats]    = useState([]);
  const [adminChatTotal, setAdminChatTotal] = useState(0);
  const [adminChatPage, setAdminChatPage] = useState(0);
  const [adminSearch,   setAdminSearch]   = useState("");
  const [historyRating, setHistoryRating] = useState("");   // "" | "5".."1" | "unrated"
  const [historyDays,   setHistoryDays]   = useState("");   // "" | "1" | "7" | "30"
  const [historyUserId, setHistoryUserId] = useState("");   // "" | a user id
  const [expandedChats, setExpandedChats] = useState(() => new Set());
  const [adminLoading,  setAdminLoading]  = useState(false);
  const ADMIN_PAGE_SIZE = 25;

  // Documents tab — every document across every user
  const [adminDocs,        setAdminDocs]        = useState(null);
  const [adminDocsLoading, setAdminDocsLoading] = useState(false);
  const [adminDocsPage,    setAdminDocsPage]    = useState(0);
  const [adminDocsSearch,  setAdminDocsSearch]  = useState("");

  // Audit Log tab — full searchable admin action history
  const [adminAuditFull,        setAdminAuditFull]        = useState(null);
  const [adminAuditLoading,     setAdminAuditLoading]     = useState(false);
  const [adminAuditPage,        setAdminAuditPage]        = useState(0);
  const [adminAuditSearch,      setAdminAuditSearch]      = useState("");
  const [adminAuditActionFilter, setAdminAuditActionFilter] = useState("");

  // personal ("My Dashboard") stats — non-admin users only, scoped to their own data
  const [meStats,        setMeStats]        = useState(null);
  const [meStatsLoading, setMeStatsLoading] = useState(false);

  // user management
  const [adminUsers,     setAdminUsers]     = useState([]);
  const [adminUsersLoading, setAdminUsersLoading] = useState(false);
  const [userSearch,     setUserSearch]     = useState("");
  const [selectedUser,   setSelectedUser]   = useState(null);   // drawer target
  const [userActivity,   setUserActivity]   = useState([]);
  const [userActivityLoading, setUserActivityLoading] = useState(false);
  const [confirmAction,  setConfirmAction]  = useState(null);   // { type, user }
  const [userActionBusy, setUserActionBusy] = useState(false);

  const fileInputRef  = useRef(null);
  const chatScrollRef = useRef(null);
  const abortRef      = useRef(null);

  const isAllScope = activeDocId === ALL_DOCS;
  const activeDoc  = docs.find(d => d.id === activeDocId) || null;

  // ── "select text in the source panel, ask about just that" ────────────────
  // selPopover: the small floating "Ask about selection" button shown right
  // after the user releases a text selection. selectedContext: what they
  // actually confirmed asking about — shown as a chip above the composer and
  // sent as `selected_text` on the next /ask call (the backend already
  // special-cases a non-empty selected_text to answer from just that excerpt
  // instead of searching the whole document — see main.py's build_context()).
  const [selPopover,      setSelPopover]      = useState(null); // { text, x, y } or null
  const [selectedContext, setSelectedContext] = useState(null); // { text, docName } or null
  const sourcePanelRef = useRef(null);

  const handleSourceMouseUp = useCallback(() => {
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : "";
    if (!text || !sourcePanelRef.current || !sel.rangeCount) { setSelPopover(null); return; }
    const anchorNode = sel.anchorNode;
    if (!anchorNode || !sourcePanelRef.current.contains(anchorNode)) { setSelPopover(null); return; }
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) { setSelPopover(null); return; }
    const wrapperRect = sourcePanelRef.current.getBoundingClientRect();
    setSelPopover({
      text: text.slice(0, 6000), // guard against pathological whole-document selections
      x: rect.left - wrapperRect.left + rect.width / 2,
      y: rect.top - wrapperRect.top,
    });
  }, []);

  const handleAskAboutSelection = useCallback(() => {
    if (!selPopover) return;
    setSelectedContext({ text: selPopover.text, docName: activeDoc?.name || "" });
    setSelPopover(null);
    window.getSelection()?.removeAllRanges();
  }, [selPopover, activeDoc]);

  useEffect(() => { setSelPopover(null); }, [activeDoc?.id, sourceOpen]);
  const task      = TASK_MODES.find(t => t.id === activeTask);
  const modelName = MODEL_LABEL[task?.model] || task?.model || "Llama 3.1 8B";

  // auto-scroll chat
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, typing]);

  // KB count on mount
  useEffect(() => {
    fetch(`${BACKEND}/knowledge_base`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) { setKbCount(data.length); setKbEntries(data); } })
      .catch(() => {});
  }, []);

  // Restore session from a stored auth token on mount
  useEffect(() => {
    if (!authToken) return;
    fetch(`${BACKEND}/me?token=${encodeURIComponent(authToken)}`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => setCurrentUser(data))
      .catch(() => { localStorage.removeItem("authToken"); setAuthToken(null); setCurrentUser(null); });
  }, [authToken]);

  // Auto-dismiss toast notifications
  useEffect(() => {
    if (!notification) return;
    const t = setTimeout(() => setNotification(""), 3500);
    return () => clearTimeout(t);
  }, [notification]);

  // Close the header account dropdown on an outside click or Escape —
  // standard menu behavior (GitHub, Slack, etc. all do this on click, not
  // actual CSS :hover, since a hover-only menu is easy to trigger by
  // accident and unusable on touch).
  useEffect(() => {
    if (!userMenuOpen) return;
    const onDown = e => { if (userMenuRef.current && !userMenuRef.current.contains(e.target)) setUserMenuOpen(false); };
    const onKey  = e => { if (e.key === "Escape") setUserMenuOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [userMenuOpen]);

  const logout = useCallback(() => {
    localStorage.removeItem("authToken");
    setAuthToken(null);
    setCurrentUser(null);
    if (mainView === "admin") setMainView("chat");
  }, [mainView]);

  const fetchMeStats = useCallback(() => {
    if (!authToken) return;
    setMeStatsLoading(true);
    fetch(`${BACKEND}/me/stats?token=${encodeURIComponent(authToken)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setMeStats(data))
      .catch(() => {})
      .finally(() => setMeStatsLoading(false));
  }, [authToken]);

  useEffect(() => {
    if (mainView === "dashboard" && currentUser) fetchMeStats();
  }, [mainView, currentUser, fetchMeStats]);

  // Admin dashboard data
  const fetchAdminData = useCallback((page = 0, filters = {}) => {
    if (!authToken) return;
    setAdminLoading(true);
    const qs = new URLSearchParams({
      token: authToken,
      limit: String(ADMIN_PAGE_SIZE),
      offset: String(page * ADMIN_PAGE_SIZE),
    });
    if (filters.search) qs.set("search", filters.search);
    if (filters.rating) qs.set("rating", filters.rating);
    if (filters.days) qs.set("since_days", filters.days);
    if (filters.userId) qs.set("user_id", filters.userId);

    Promise.all([
      fetch(`${BACKEND}/admin/overview?token=${encodeURIComponent(authToken)}`).then(r => r.ok ? r.json() : null),
      fetch(`${BACKEND}/admin/chat_history?${qs.toString()}`).then(r => r.ok ? r.json() : null),
      fetch(`${BACKEND}/admin/analytics?token=${encodeURIComponent(authToken)}`).then(r => r.ok ? r.json() : null),
      fetch(`${BACKEND}/admin/audit_log?token=${encodeURIComponent(authToken)}`).then(r => r.ok ? r.json() : null),
      fetch(`${BACKEND}/admin/system_status?token=${encodeURIComponent(authToken)}`).then(r => r.ok ? r.json() : null),
    ])
      .then(([overview, history, analytics, auditLog, systemStatus]) => {
        if (overview) setAdminOverview(overview);
        if (history) { setAdminChats(history.items || []); setAdminChatTotal(history.total_count || 0); }
        if (analytics) setAdminAnalytics(analytics);
        if (auditLog) setAdminAuditLog(auditLog.items || []);
        if (systemStatus) setAdminSystemStatus(systemStatus);
      })
      .catch(() => {})
      .finally(() => setAdminLoading(false));
  }, [authToken]);

  useEffect(() => {
    if (mainView === "admin" && currentUser?.is_admin) {
      fetchAdminData(adminChatPage, { search: adminSearch, rating: historyRating, days: historyDays, userId: historyUserId });
    }
  }, [mainView, currentUser, adminChatPage, adminSearch, historyRating, historyDays, historyUserId, fetchAdminData]);

  // Document Insights — extracted facts/keywords/entities/doc-type per document
  const fetchAdminInsights = useCallback(() => {
    if (!authToken) return;
    setAdminInsightsLoading(true);
    fetch(`${BACKEND}/admin/document_insights?token=${encodeURIComponent(authToken)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setAdminInsights(data); })
      .catch(() => {})
      .finally(() => setAdminInsightsLoading(false));
  }, [authToken]);

  useEffect(() => {
    if (mainView === "admin" && adminTab === "insights" && currentUser?.is_admin) fetchAdminInsights();
  }, [mainView, adminTab, currentUser, fetchAdminInsights]);

  const analyzePendingDocuments = useCallback(() => {
    if (!authToken) return;
    setInsightsAnalyzeBusy(true);
    fetch(`${BACKEND}/admin/document_insights/analyze_pending?token=${encodeURIComponent(authToken)}&max_docs=15`, { method: "POST" })
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        const n = data?.queued || 0;
        setNotification(n > 0 ? `Analyzing ${n} document${n === 1 ? "" : "s"}… check back in a minute.` : "No documents pending analysis.");
        // extraction runs in the background on the server; poll a couple of times to pick up results
        [8000, 20000, 40000].forEach(delay => setTimeout(fetchAdminInsights, delay));
      })
      .catch(() => setNotification("Could not start analysis."))
      .finally(() => setInsightsAnalyzeBusy(false));
  }, [authToken, fetchAdminInsights]);

  // Documents tab — every document across every user
  const fetchAdminDocuments = useCallback((page = 0, search = "") => {
    if (!authToken) return;
    setAdminDocsLoading(true);
    const qs = new URLSearchParams({ token: authToken, limit: String(ADMIN_PAGE_SIZE), offset: String(page * ADMIN_PAGE_SIZE) });
    if (search) qs.set("search", search);
    fetch(`${BACKEND}/admin/documents?${qs.toString()}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setAdminDocs(data); })
      .catch(() => {})
      .finally(() => setAdminDocsLoading(false));
  }, [authToken]);

  useEffect(() => {
    if (mainView === "admin" && adminTab === "documents" && currentUser?.is_admin) {
      fetchAdminDocuments(adminDocsPage, adminDocsSearch);
    }
  }, [mainView, adminTab, currentUser, adminDocsPage, adminDocsSearch, fetchAdminDocuments]);

  // Audit Log tab — full searchable admin action history
  const fetchAuditLogFull = useCallback((page = 0, search = "", action = "") => {
    if (!authToken) return;
    setAdminAuditLoading(true);
    const qs = new URLSearchParams({ token: authToken, limit: String(ADMIN_PAGE_SIZE), offset: String(page * ADMIN_PAGE_SIZE) });
    if (search) qs.set("search", search);
    if (action) qs.set("action", action);
    fetch(`${BACKEND}/admin/audit_log?${qs.toString()}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setAdminAuditFull(data); })
      .catch(() => {})
      .finally(() => setAdminAuditLoading(false));
  }, [authToken]);

  useEffect(() => {
    if (mainView === "admin" && adminTab === "audit" && currentUser?.is_admin) {
      fetchAuditLogFull(adminAuditPage, adminAuditSearch, adminAuditActionFilter);
    }
  }, [mainView, adminTab, currentUser, adminAuditPage, adminAuditSearch, adminAuditActionFilter, fetchAuditLogFull]);

  // User management — list, drill-down activity, and admin actions
  const fetchAdminUsers = useCallback(() => {
    if (!authToken) return;
    setAdminUsersLoading(true);
    fetch(`${BACKEND}/admin/users?token=${encodeURIComponent(authToken)}`)
      .then(r => r.ok ? r.json() : [])
      .then(data => setAdminUsers(Array.isArray(data) ? data : []))
      .catch(() => {})
      .finally(() => setAdminUsersLoading(false));
  }, [authToken]);

  useEffect(() => {
    if (mainView === "admin" && currentUser?.is_admin) fetchAdminUsers();
  }, [mainView, currentUser, fetchAdminUsers]);

  const openUserDetail = useCallback((user) => {
    setSelectedUser(user);
    setUserActivity([]);
    setUserActivityLoading(true);
    fetch(`${BACKEND}/admin/chat_history?token=${encodeURIComponent(authToken)}&limit=8&user_id=${encodeURIComponent(user.id)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setUserActivity(data?.items || []))
      .catch(() => {})
      .finally(() => setUserActivityLoading(false));
  }, [authToken]);

  const updateUser = useCallback(async (userId, patch) => {
    setUserActionBusy(true);
    try {
      const res = await fetch(`${BACKEND}/admin/users/${userId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: authToken, ...patch }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setNotification(err.detail || "Could not update user");
        return false;
      }
      setAdminUsers(prev => prev.map(u => u.id === userId ? { ...u, ...patch } : u));
      setSelectedUser(prev => prev && prev.id === userId ? { ...prev, ...patch } : prev);
      return true;
    } catch {
      setNotification("Could not reach the server");
      return false;
    } finally {
      setUserActionBusy(false);
    }
  }, [authToken]);

  const deleteUser = useCallback(async (userId) => {
    setUserActionBusy(true);
    try {
      const res = await fetch(`${BACKEND}/admin/users/${userId}?token=${encodeURIComponent(authToken)}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setNotification(err.detail || "Could not delete user");
        return false;
      }
      setAdminUsers(prev => prev.filter(u => u.id !== userId));
      setSelectedUser(prev => prev && prev.id === userId ? null : prev);
      setNotification("User deleted");
      return true;
    } catch {
      setNotification("Could not reach the server");
      return false;
    } finally {
      setUserActionBusy(false);
    }
  }, [authToken]);

  // Fetch KB entries when tab opened
  const fetchKbEntries = useCallback(() => {
    setKbLoading(true);
    fetch(`${BACKEND}/knowledge_base`)
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data)) {
          setKbEntries(data);
          setKbCount(data.length);
        }
      })
      .catch(() => {})
      .finally(() => setKbLoading(false));
  }, []);

  // Delete KB entry
  const deleteKbEntry = useCallback((id) => {
    fetch(`${BACKEND}/knowledge_base/${id}?token=${encodeURIComponent(authToken || "")}`, { method: "DELETE" })
      .then(() => fetchKbEntries())
      .catch(() => {});
  }, [authToken, fetchKbEntries]);

  // Pull an external web page into the KB — the server fetches it, extracts
  // readable text, and generates Q&A pairs from it just like a document upload.
  const importKbFromUrl = useCallback(async () => {
    if (!kbUrlInput.trim() || kbUrlBusy) return;
    setKbUrlBusy(true); setKbUrlMsg(null);
    try {
      const res = await fetch(`${BACKEND}/knowledge_base/import_url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: kbUrlInput.trim(), token: authToken || null }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Import failed");
      setKbUrlMsg({ ok: true, text: data.message || "Imported." });
      setKbUrlInput("");
      fetchKbEntries();
    } catch (e) {
      setKbUrlMsg({ ok: false, text: e.message || "Could not import that URL." });
    } finally {
      setKbUrlBusy(false);
    }
  }, [kbUrlInput, kbUrlBusy, authToken, fetchKbEntries]);

  const addManualKbEntry = useCallback(async () => {
    if (!kbManualQ.trim() || !kbManualA.trim() || kbManualBusy) return;
    setKbManualBusy(true); setKbManualMsg(null);
    try {
      const res = await fetch(`${BACKEND}/knowledge_base`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: [{ question: kbManualQ.trim(), answer: kbManualA.trim(), source: kbManualSrc.trim() || null }],
          source_type: "manual",
          token: authToken || null,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Could not add that entry");
      setKbManualMsg({ ok: true, text: "Entry added." });
      setKbManualQ(""); setKbManualA(""); setKbManualSrc("");
      fetchKbEntries();
    } catch (e) {
      setKbManualMsg({ ok: false, text: e.message || "Could not add that entry." });
    } finally {
      setKbManualBusy(false);
    }
  }, [kbManualQ, kbManualA, kbManualSrc, kbManualBusy, authToken, fetchKbEntries]);

  // Import a batch of Q&A pairs from a JSON array ([{question,answer,source?}])
  // or a CSV with question,answer,source columns — parsed entirely client-side.
  const handleKbImportFile = useCallback(async (file) => {
    if (!file || kbImportBusy) return;
    setKbImportBusy(true); setKbImportMsg(null);
    try {
      const text = await file.text();
      let items = [];
      if (file.name.toLowerCase().endsWith(".json")) {
        const parsed = JSON.parse(text);
        const arr = Array.isArray(parsed) ? parsed : parsed.items;
        if (!Array.isArray(arr)) throw new Error("JSON must be an array of {question, answer} objects");
        items = arr.map(it => ({ question: String(it.question || "").trim(), answer: String(it.answer || "").trim(), source: it.source ? String(it.source) : file.name }));
      } else {
        const lines = text.split(/\r?\n/).filter(l => l.trim());
        const header = lines[0].split(",").map(h => h.trim().toLowerCase());
        const qIdx = header.indexOf("question"), aIdx = header.indexOf("answer"), sIdx = header.indexOf("source");
        if (qIdx === -1 || aIdx === -1) throw new Error('CSV needs "question" and "answer" columns');
        const parseCsvLine = line => {
          const cells = []; let cur = "", inQ = false;
          for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') { if (inQ && line[i + 1] === '"') { cur += '"'; i++; } else inQ = !inQ; }
            else if (ch === "," && !inQ) { cells.push(cur); cur = ""; }
            else cur += ch;
          }
          cells.push(cur);
          return cells;
        };
        items = lines.slice(1).map(line => {
          const cells = parseCsvLine(line);
          return { question: (cells[qIdx] || "").trim(), answer: (cells[aIdx] || "").trim(), source: sIdx !== -1 ? (cells[sIdx] || "").trim() : file.name };
        });
      }
      items = items.filter(it => it.question && it.answer);
      if (items.length === 0) throw new Error("No valid question/answer rows found in that file");

      const res = await fetch(`${BACKEND}/knowledge_base`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items, source_type: "external_import", token: authToken || null }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Import failed");
      setKbImportMsg({ ok: true, text: `${data.added ?? items.length} entries imported from ${file.name}.` });
      fetchKbEntries();
    } catch (e) {
      setKbImportMsg({ ok: false, text: e.message || "Could not import that file." });
    } finally {
      setKbImportBusy(false);
      if (kbImportFileRef.current) kbImportFileRef.current.value = "";
    }
  }, [kbImportBusy, authToken, fetchKbEntries]);

  // CourtListener search — called after document upload
  const searchPrecedents = useCallback(async (docText, docName) => {
    setPrecedents([]);
    setPrecLoading(true);
    const query = [
      docName.replace(/\.(pdf|docx|txt)$/i, "").replace(/[_-]/g, " "),
      (docText || "").slice(0, 200),
    ].join(" ").trim().slice(0, 100);
    const results = await courtListenerQuery(query, 5);
    setPrecedents(results.map(r => ({ ...r, score: null })));
    setPrecLoading(false);
  }, []);

  const handleCitation = useCallback((page) => {
    setHlPage(page);
    setSourceOpen(true);
  }, []);

  // Thumbs up/down on an assistant reply — persists to chat_history.rating via the
  // existing (previously unwired) /put_ratings endpoint, feeding the admin dashboard's
  // average-rating stat.
  const submitRating = useCallback((msgId, entryId, rating) => {
    setMessages(prev => prev.map(m => m.id === msgId ? { ...m, userRating: rating } : m));
    fetch(`${BACKEND}/put_ratings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: entryId, rating, comment: "" }),
    }).catch(() => {
      setMessages(prev => prev.map(m => m.id === msgId ? { ...m, userRating: null } : m));
    });
  }, []);

  // Same /put_ratings call, but for a chat row shown on "My Dashboard" (not
  // part of the live chat thread) — updates the local meStats snapshot
  // optimistically instead of the messages array.
  const submitDashboardRating = useCallback((chatId, rating) => {
    setMeStats(prev => prev ? {
      ...prev,
      recent_chats: prev.recent_chats.map(c => c.id === chatId ? { ...c, rating } : c),
      rated_count: prev.recent_chats.find(c => c.id === chatId)?.rating ? prev.rated_count : prev.rated_count + 1,
    } : prev);
    fetch(`${BACKEND}/put_ratings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: chatId, rating, comment: "" }),
    }).catch(() => setNotification("Could not save rating"));
  }, []);

  // ── upload ────────────────────────────────────────────────────────────────
  const processFiles = useCallback(async (files) => {
    const allowed = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain",
    ];
    const valid = Array.from(files).filter(f => {
      if (f.size > 20 * 1024 * 1024) { setUploadErr(`${f.name} exceeds 20MB limit`); return false; }
      if (!allowed.includes(f.type))  { setUploadErr(`${f.name}: unsupported type (PDF, DOCX, TXT only)`); return false; }
      return true;
    });
    if (!valid.length) return;

    setUploading(true);
    setUploadErr("");

    const addedDocs = [];

    for (const file of valid) {
      try {
        let parsed;
        if (file.type === "application/pdf") {
          parsed = await parsePDF(file);
        } else if (file.type.includes("wordprocessingml")) {
          parsed = await parseDOCX(file);
        } else {
          parsed = await parseTXT(file);
        }

        // Send extracted text to backend
        const fd  = new FormData();
        const blob = new Blob([parsed.text], { type: "text/plain" });
        fd.append("files", blob, file.name);
        if (authToken) fd.append("auth_token", authToken);

        let docId = null;
        try {
          const res = await fetch(`${BACKEND}/upload`, { method: "POST", body: fd });
          if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data) && data[0]?.document_id) docId = data[0].document_id;
          }
        } catch (_) {}

        const docType = /contract|agreement/i.test(file.name) ? "contract"
          : /code|statute|regulation/i.test(file.name) ? "statute" : "case";

        const newDoc = {
          id:                docId || Date.now(),
          name:              file.name,
          text:              parsed.text,
          pages:             parsed.pages,
          isPdf:             parsed.isPdf,
          isDocx:            parsed.isDocx            || false,
          htmlPages:         parsed.htmlPages          || null,
          html:              parsed.html               || null,
          arrayBuffer:       parsed.arrayBuffer        || null,
          convertedFromDocx: parsed.convertedFromDocx  || false,
          type:              docType,
        };

        addedDocs.push(newDoc);
        setDocs(prev => [...prev, newDoc]);
        setHlPage(null);
        setSuggestions([]);
        setSourceOpen(true);
        setMainView("chat");
        setMessages(m => [...m, {
          id: Date.now(), role: "assistant", model: task?.model,
          text: `"${file.name}" processed — ${parsed.pages} page${parsed.pages !== 1 ? "s" : ""} indexed. Ask a question or click a suggested question below.`,
          citations: [],
        }]);

        // Generate suggested questions
        fetchSuggestions(parsed.text.slice(0, 2500), file.name, task?.model || "llama3.1:8b");
        searchPrecedents(parsed.text.slice(0, 300), file.name);

      } catch (err) {
        setUploadErr(`Error processing ${file.name}: ${err.message}`);
      }
    }

    // Once 2+ documents exist in total, default to combined "All Documents" scope
    // so the very next question is answered from everything uploaded, not just
    // whichever file happened to be processed last.
    if (addedDocs.length) {
      const totalCount = docs.length + addedDocs.length;
      setActiveDocId(totalCount > 1 ? ALL_DOCS : addedDocs[addedDocs.length - 1].id);
    }

    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [task, docs.length, authToken]);

  async function fetchSuggestions(snippet, filename, model) {
    setSuggestions([]);

    // If no usable text, skip the LLM call entirely
    const cleanSnippet = (snippet || "").replace(/\s+/g, " ").trim();
    if (cleanSnippet.length < 100) {
      setSuggestions([
        "What are the key parties and their obligations?",
        "What are the main legal risks in this document?",
        "What are the important dates and deadlines?",
        "Are there any unusual or concerning clauses?",
      ]);
      return;
    }

    try {
      const res = await fetch(`${BACKEND}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: `Generate exactly 4 specific questions a lawyer would ask about this document. Return ONLY a JSON array of 4 strings — no explanation, no markdown, no extra text.\n\nDocument excerpt:\n${cleanSnippet.slice(0, 2000)}`,
          model,
          active_document_name: filename,
          selected_text: "",
          selections: [],
        }),
      });
      if (!res.ok) throw new Error();
      const reader = res.body.getReader();
      const dec    = new TextDecoder();
      let   full   = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value);
        if (/^__(SEARCHING|READY|CONTEXT|ENTRY_ID)/.test(chunk.trim())) continue;
        full += chunk;
      }
      const match = full.match(/\[[\s\S]*?\]/);
      if (match) {
        const arr = JSON.parse(match[0]);
        if (Array.isArray(arr) && arr.length > 0) {
          setSuggestions(arr.filter(s => typeof s === "string" && s.length > 10).slice(0, 4));
          return;
        }
      }
      throw new Error("no valid array");
    } catch {
      setSuggestions([
        "What are the key parties and their obligations?",
        "What are the main legal risks in this document?",
        "What are the important dates and deadlines?",
        "Are there any unusual or concerning clauses?",
      ]);
    }
  }

  // ── ask / stream ──────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text, taskOverride) => {
    if (!text.trim() || typing) return;

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // taskOverride lets a caller (e.g. clicking a Task Mode button) fire a message
    // for the mode it's switching TO, without waiting a render for activeTask to
    // actually update — reading activeTask here would still see the old mode.
    const effectiveTaskId = taskOverride || activeTask;
    const effectiveTask   = TASK_MODES.find(t => t.id === effectiveTaskId) || task;

    const scope = isAllScope
      ? { type: "all", count: docs.length }
      : activeDoc ? { type: "doc", name: activeDoc.name } : null;
    // Consume the pending selection (if any) for this one message, then clear
    // it — it shouldn't silently carry over and get attached to a later,
    // unrelated question.
    const askedContext = selectedContext;
    const userMsg = { id: Date.now(), role: "user", text, scope, context: askedContext };
    setMessages(m => [...m, userMsg]);
    setInput("");
    setSelectedContext(null);
    setTyping(true);

    const prefix = effectiveTaskId !== "research" ? TASK_PROMPTS[effectiveTaskId] + "\n\n" : "";
    const msgId  = Date.now() + 1;
    setMessages(m => [...m, { id: msgId, role: "assistant", model: effectiveTask?.model, text: "", citations: [], streaming: true }]);

    // Combined mode sends the exact text of every doc the user uploaded in this
    // session, so answers are scoped to those documents only — not the shared,
    // unfiltered vector index, which can contain unrelated documents other users uploaded.
    const perDocLimit = Math.max(2000, Math.min(8000, Math.floor(16000 / Math.max(docs.length, 1))));
    const combinedSelections = isAllScope
      ? docs.map(d => ({
          id:            String(d.id),
          document_name: d.name,
          text:          (d.text || "").slice(0, perDocLimit) || `(No extractable text for ${d.name})`,
        }))
      : [];

    try {
      const res = await fetch(`${BACKEND}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          question:             prefix + text,
          model:                effectiveTask?.model || "llama3.1:8b",
          active_document_name: isAllScope ? null : (activeDoc?.name || null),
          selected_text:        askedContext?.text || "",
          selections:           combinedSelections,
          auth_token:           authToken || null,
        }),
      });
      if (!res.ok) throw new Error(`${res.status}`);

      const reader = res.body.getReader();
      const dec    = new TextDecoder();
      let   full   = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const raw = dec.decode(value);
        if (raw.includes("__ENTRY_ID__")) {
          const m = raw.match(/__ENTRY_ID__(\d+)__/);
          if (m) setMessages(prev => prev.map(msg =>
            msg.id === msgId ? { ...msg, entryId: parseInt(m[1]) } : msg
          ));
          continue;
        }
        if (/^__(SEARCHING|READY|CONTEXT)/.test(raw.trim())) continue;
        if (raw.startsWith("__ERROR__")) {
          const detail = raw.match(/^__ERROR__(.*?)__?$/s)?.[1]?.trim();
          setMessages(prev => prev.map(msg =>
            msg.id === msgId ? { ...msg, text: detail || "Backend error — check that Ollama is running.", streaming: false } : msg
          ));
          setTyping(false);
          return;
        }
        full += raw;
        setMessages(prev => prev.map(msg => msg.id === msgId ? { ...msg, text: full } : msg));
      }

      // Extract page citations from answer text
      const refs = [...full.matchAll(/\b(?:page|p\.)\s*(\d+)\b/gi)]
        .map(m => ({ page: parseInt(m[1]), label: `p.${m[1]}` }))
        .filter((v, i, a) => a.findIndex(x => x.page === v.page) === i)
        .slice(0, 5);

      setMessages(prev => prev.map(msg =>
        msg.id === msgId ? { ...msg, text: full, citations: refs, streaming: false } : msg
      ));
      if (refs.length) handleCitation(refs[0].page);
      setKbCount(k => k + 1);

    } catch (err) {
      if (err.name !== "AbortError") {
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, text: `Could not reach backend (${err.message}). Is the server running at ${BACKEND}?`, streaming: false }
            : msg
        ));
      }
    }
    setTyping(false);
  }, [activeTask, activeDoc, isAllScope, docs.length, task, typing, handleCitation, authToken, selectedContext]);

  // ── task-mode tools — non-streaming, structured-output calls that reuse the
  // same /ask + document-context machinery as chat, but ask the model for JSON
  // and render the parsed result as a purpose-built view instead of a chat bubble.
  const runToolQuery = useCallback(async (promptText, modelOverride) => {
    const perDocLimit = Math.max(2000, Math.min(8000, Math.floor(16000 / Math.max(docs.length, 1))));
    const combinedSelections = isAllScope
      ? docs.map(d => ({
          id: String(d.id), document_name: d.name,
          text: (d.text || "").slice(0, perDocLimit) || `(No extractable text for ${d.name})`,
        }))
      : [];
    const res = await fetch(`${BACKEND}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question:             promptText,
        model:                modelOverride || task?.model || "llama3.1:8b",
        active_document_name: isAllScope ? null : (activeDoc?.name || null),
        selected_text:        "",
        selections:           combinedSelections,
        auth_token:           authToken || null,
      }),
    });
    if (!res.ok) throw new Error(`Server error (${res.status})`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let full = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const raw = dec.decode(value);
      if (raw.includes("__ENTRY_ID__")) continue;
      if (/^__(SEARCHING|READY|CONTEXT)/.test(raw.trim())) continue;
      if (raw.startsWith("__ERROR__")) {
        throw new Error(raw.replace(/^__ERROR__/, "").replace(/__$/, "").trim() || "Backend error");
      }
      full += raw;
    }
    return full;
  }, [docs, isAllScope, activeDoc, task, authToken]);

  const runArgumentTool = useCallback(async (claim) => {
    if (!claim.trim() || toolLoading) return;
    setToolLoading(true); setToolError("");
    try {
      const prompt = `You are a legal research assistant building a structured legal argument.\n\nClaim to argue: "${claim}"\n\nReturn ONLY a JSON object with this exact shape, no markdown, no explanation outside the JSON:\n{"issue": "...", "rule": "...", "application": "...", "conclusion": "...", "weaknesses": ["...", "..."]}\nCite specific page numbers inline within the text fields wherever the document supports a claim. If the document doesn't address the claim, say so honestly in "application" rather than inventing facts.`;
      const full = await runToolQuery(prompt);
      const data = extractJson(full, "object");
      const key = activeDocId;
      const entry = { id: Date.now(), claim, raw: full, parsed: !!data, ...(data || {}) };
      setArgumentByDoc(prev => ({ ...prev, [key]: [entry, ...(prev[key] || [])] }));
      setArgumentClaim("");
    } catch (e) {
      setToolError(e.message || "Could not generate the argument.");
    } finally {
      setToolLoading(false);
    }
  }, [runToolQuery, activeDocId, toolLoading]);

  const runRiskTool = useCallback(async () => {
    if (toolLoading) return;
    setToolLoading(true); setToolError("");
    try {
      const prompt = `You are a legal risk analyst. Scan the document for risk factors: ambiguous language, missing standard clauses, unfavorable terms, and jurisdiction risks.\n\nReturn ONLY a JSON array, no markdown, no explanation outside the JSON, of up to 8 objects with this exact shape:\n[{"title": "...", "severity": "High", "description": "...", "recommendation": "...", "location": "e.g. page 3 or clause name"}]\nseverity must be exactly "High", "Medium", or "Low". If the document has no notable risks, return [].`;
      const full = await runToolQuery(prompt);
      const data = extractJson(full, "array");
      setRiskByDoc(prev => ({ ...prev, [activeDocId]: { items: Array.isArray(data) ? data : null, raw: full, ranAt: new Date().toISOString() } }));
    } catch (e) {
      setToolError(e.message || "Could not run the risk scan.");
    } finally {
      setToolLoading(false);
    }
  }, [runToolQuery, activeDocId, toolLoading]);

  const runClauseTool = useCallback(async (clauseType) => {
    if (!clauseType.trim() || toolLoading) return;
    setToolLoading(true); setToolError("");
    try {
      const prompt = `You are a contract analysis assistant. Extract the "${clauseType}" clause from the document.\n\nReturn ONLY a JSON object with this exact shape, no markdown, no explanation outside the JSON:\n{"found": true, "exact_text": "...", "explanation": "plain English explanation", "risk_level": "Low"}\nrisk_level must be exactly "Low", "Medium", or "High". If the clause is not present, set "found" to false, leave "exact_text" empty, and explain what's missing in "explanation".`;
      const full = await runToolQuery(prompt);
      const data = extractJson(full, "object");
      const key = activeDocId;
      const entry = { id: Date.now(), clauseType, raw: full, parsed: !!data, ...(data || {}) };
      setClauseByDoc(prev => ({ ...prev, [key]: [entry, ...(prev[key] || [])] }));
      setCustomClauseInput("");
    } catch (e) {
      setToolError(e.message || "Could not extract that clause.");
    } finally {
      setToolLoading(false);
    }
  }, [runToolQuery, activeDocId, toolLoading]);

  const runSummaryTool = useCallback(async (length) => {
    if (toolLoading) return;
    setToolLoading(true); setToolError("");
    const lengthInstruction = { brief: "very brief (2-3 sentences per section)", standard: "standard", detailed: "thorough and detailed" }[length] || "standard";
    try {
      const prompt = `You are a legal document analyst. Produce a ${lengthInstruction} structured summary of the document.\n\nReturn ONLY a JSON object with this exact shape, no markdown, no explanation outside the JSON:\n{"parties": ["..."], "key_dates": [{"date": "...", "description": "..."}], "obligations": ["..."], "notable_clauses": ["..."]}\nIf a section doesn't apply, return an empty array for it.`;
      const full = await runToolQuery(prompt);
      const data = extractJson(full, "object");
      setSummaryByDoc(prev => ({ ...prev, [activeDocId]: { ...(data || {}), raw: full, parsed: !!data, length, ranAt: new Date().toISOString() } }));
    } catch (e) {
      setToolError(e.message || "Could not generate the summary.");
    } finally {
      setToolLoading(false);
    }
  }, [runToolQuery, activeDocId, toolLoading]);

  const runPrecedentsTool = useCallback(async () => {
    if (toolLoading) return;
    setToolLoading(true); setToolError("");
    try {
      const prompt = `You are a legal research assistant. Identify the 3 to 5 key legal issues or causes of action raised in this document that a lawyer would want case law on.\n\nReturn ONLY a JSON array of short issue strings, no markdown, no explanation outside the JSON:\n["issue 1", "issue 2"]`;
      const full = await runToolQuery(prompt);
      const issues = extractJson(full, "array");
      if (!Array.isArray(issues) || issues.length === 0) {
        setPrecedentsToolByDoc(prev => ({ ...prev, [activeDocId]: { issues: [], raw: full, ranAt: new Date().toISOString() } }));
        return;
      }
      const perIssue = await Promise.all(
        issues.slice(0, 5).map(async issue => ({ issue, cases: await courtListenerQuery(String(issue).slice(0, 100), 3) }))
      );
      setPrecedentsToolByDoc(prev => ({ ...prev, [activeDocId]: { issues: perIssue, ranAt: new Date().toISOString() } }));
    } catch (e) {
      setToolError(e.message || "Could not search for precedents.");
    } finally {
      setToolLoading(false);
    }
  }, [runToolQuery, activeDocId, toolLoading]);

  const exportChat = useCallback(() => {
    if (!messages.length) { alert("No conversation to export yet."); return; }
    const content = messages.map(m =>
      `[${m.role === "user" ? "You" : "Assistant"}]\n${m.text}\n`
    ).join("\n---\n\n");
    const blob = new Blob([content], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `SynergeReader_Export_${Date.now()}.txt`;
    a.click(); URL.revokeObjectURL(url);
  }, [messages]);

  // drag-and-drop (works anywhere in the app, like a modern chat client)
  const onDragOver  = e => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = e => { e.preventDefault(); setIsDragging(false); };
  const onDrop      = e => { e.preventDefault(); setIsDragging(false); processFiles(e.dataTransfer.files); };

  const hasDocs = docs.length > 0;

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div
      onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
      style={{
        display: "flex", height: "100vh", width: "100%",
        fontFamily: UI_FONT, background: "#fff", color: "#0d0d0d",
        overflow: "hidden", position: "relative",
      }}
    >
      {/* full-screen drag overlay */}
      {isDragging && (
        <div style={{
          position: "absolute", inset: 0, zIndex: 50,
          background: "rgba(37,99,235,.08)",
          border: "3px dashed #2563eb",
          display: "flex", alignItems: "center", justifyContent: "center",
          pointerEvents: "none",
        }}>
          <div style={{
            background: "#fff", border: "1px solid #dbeafe", borderRadius: "12px",
            padding: "24px 36px", boxShadow: "0 8px 30px rgba(0,0,0,.12)",
            fontSize: "15px", fontWeight: 600, color: "#1d4ed8",
          }}>Drop PDF, DOCX or TXT to upload</div>
        </div>
      )}

      <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" multiple
        style={{ display: "none" }} onChange={e => processFiles(e.target.files)} />

      {/* ── SIDEBAR ────────────────────────────────────────────────────── */}
      <div style={{
        width: sidebarOpen ? "268px" : "0px",
        flexShrink: 0, overflow: "hidden",
        background: "#13151a", color: "#ececec",
        display: "flex", flexDirection: "column",
        transition: "width .18s ease",
        borderRight: "1px solid #1e212a",
      }}>
        <div style={{ width: "268px", height: "100%", display: "flex", flexDirection: "column" }}>

          {/* brand + collapse */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "16px 14px 14px",
            borderBottom: "1px solid #1e212a",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "9px" }}>
              <div style={{
                width: "28px", height: "28px",
                background: "linear-gradient(135deg,#3b82f6,#1d4ed8)",
                borderRadius: "8px", display: "flex", alignItems: "center",
                justifyContent: "center", color: "#fff",
                boxShadow: "0 2px 8px rgba(59,130,246,.35)",
              }}><IconScale width={15} height={15} /></div>
              <span style={{ fontWeight: 700, fontSize: "14.5px", color: "#fff", letterSpacing: "-.01em" }}>SynergeReader</span>
            </div>
            <button onClick={() => setSidebarOpen(false)} title="Collapse sidebar" style={{
              background: "none", border: "none", color: "#6b7080", cursor: "pointer",
              padding: "6px", borderRadius: "7px", display: "flex", transition: "background .12s, color .12s",
            }}
              onMouseEnter={e => { e.currentTarget.style.background = "#1c1f28"; e.currentTarget.style.color = "#ececec"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#6b7080"; }}
            ><IconPanel width={15} height={15} /></button>
          </div>

          {/* new upload */}
          <div style={{ padding: "14px 14px 6px" }}>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: "9px",
                background: "linear-gradient(180deg,#2563eb,#1d4ed8)", border: "none",
                color: "#fff", padding: "10px 13px", borderRadius: "10px",
                cursor: uploading ? "default" : "pointer", fontSize: "13px",
                fontFamily: UI_FONT, fontWeight: 600, transition: "filter .12s, transform .08s",
                boxShadow: "0 2px 10px rgba(37,99,235,.28)",
              }}
              onMouseEnter={e => e.currentTarget.style.filter = "brightness(1.08)"}
              onMouseLeave={e => e.currentTarget.style.filter = "brightness(1)"}
            ><IconPlus width={16} height={16} />{uploading ? "Processing…" : "Upload Document"}</button>
            {uploadErr && (
              <div style={{
                fontSize: "10.5px", color: "#fca5a5",
                background: "#2c1618", border: "1px solid #4c2226",
                borderRadius: "8px", padding: "7px 9px", marginTop: "8px", lineHeight: "1.45",
              }}>{uploadErr}</div>
            )}
          </div>

          {/* documents list */}
          <div style={{ padding: "10px 14px 6px" }}>
            <div style={{
              fontSize: "10px", color: "#5b6072", letterSpacing: ".07em",
              textTransform: "uppercase", fontWeight: 700,
            }}>Documents</div>
          </div>
          <div style={{ flex: hasDocs ? "0 1 auto" : "0 0 auto", maxHeight: "32vh", overflow: "auto", padding: "0 10px" }}>
            {!hasDocs && (
              <div style={{ padding: "8px 6px", fontSize: "11.5px", color: "#565b6c", lineHeight: "1.5" }}>
                No documents yet. Upload one to start chatting.
              </div>
            )}
            {docs.length > 1 && (
              <div
                onClick={() => { setActiveDocId(ALL_DOCS); setHlPage(null); setMainView("chat"); }}
                title="Ask questions across every uploaded document at once"
                style={{
                  display: "flex", alignItems: "center", gap: "9px",
                  padding: "8px 9px", borderRadius: "9px", marginBottom: "8px",
                  cursor: "pointer",
                  background: isAllScope ? "rgba(59,130,246,.22)" : "rgba(59,130,246,.09)",
                  borderLeft: isAllScope ? "3px solid #3b82f6" : "3px solid rgba(59,130,246,.35)",
                  transition: "background .12s",
                }}
                onMouseEnter={e => { if (!isAllScope) e.currentTarget.style.background = "rgba(59,130,246,.15)"; }}
                onMouseLeave={e => { if (!isAllScope) e.currentTarget.style.background = "rgba(59,130,246,.09)"; }}
              >
                <div style={{
                  width: "24px", height: "24px", borderRadius: "7px", flexShrink: 0,
                  background: "rgba(59,130,246,.25)", color: "#93c5fd",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}><IconLayers width={13} height={13} /></div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{
                    fontSize: "12.5px", lineHeight: "1.3",
                    color: isAllScope ? "#fff" : "#dbe4ff",
                    fontWeight: 600,
                  }}>All Documents</div>
                  <div style={{ fontSize: "10px", color: "#7d8ac2" }}>Combined · {docs.length} docs</div>
                </div>
              </div>
            )}
            {docs.map(doc => {
              const active = doc.id === activeDocId;
              const typeColor = DOC_TYPE_COLOR[doc.type] || "#6b7080";
              return (
                <div key={doc.id}
                  onClick={() => { setActiveDocId(doc.id); setHlPage(null); setMainView("chat"); }}
                  style={{
                    display: "flex", alignItems: "center", gap: "9px",
                    padding: "7px 9px", borderRadius: "9px", marginBottom: "2px",
                    cursor: "pointer",
                    background: active ? "#1d2029" : "transparent",
                    borderLeft: active ? `3px solid ${typeColor}` : "3px solid transparent",
                    transition: "background .1s",
                  }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = "#181a21"; }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = "transparent"; }}
                >
                  <div style={{
                    width: "24px", height: "24px", borderRadius: "7px", flexShrink: 0,
                    background: typeColor + "26", color: typeColor,
                    display: "flex", alignItems: "center", justifyContent: "center",
                  }}><IconFile width={13} height={13} /></div>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{
                      fontSize: "12.5px", lineHeight: "1.3",
                      color: active ? "#fff" : "#c6cad6",
                      fontWeight: active ? 600 : 400,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>{doc.name}</div>
                    <div style={{ fontSize: "10px", color: "#5b6072" }}>
                      {doc.pages} page{doc.pages !== 1 ? "s" : ""} · {doc.type}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* task mode */}
          <div style={{ padding: "14px 14px 6px", borderTop: "1px solid #1e212a", marginTop: "10px" }}>
            <div style={{
              fontSize: "10px", color: "#5b6072", letterSpacing: ".07em",
              textTransform: "uppercase", fontWeight: 700,
            }}>Task Mode</div>
          </div>
          <div style={{ padding: "0 10px" }}>
            {TASK_MODES.map(t => {
              const active = activeTask === t.id;
              return (
                <button key={t.id} onClick={() => {
                  setActiveTask(t.id);
                  setMainView("chat");
                  setToolError("");
                  // Modes that need no user input run themselves the moment you
                  // switch in, so picking a mode is never a dead click — but
                  // only once per document, so re-clicking doesn't burn a call.
                  if (hasDocs) {
                    if (t.id === "risk" && !riskByDoc[activeDocId]) runRiskTool();
                    if (t.id === "summarize" && !summaryByDoc[activeDocId]) runSummaryTool(summaryLength);
                    if (t.id === "precedents" && !precedentsToolByDoc[activeDocId]) runPrecedentsTool();
                  }
                }} style={{
                  display: "flex", alignItems: "center", gap: "9px",
                  width: "100%", background: active ? t.color + "1f" : "none",
                  border: "none", borderLeft: active ? `3px solid ${t.color}` : "3px solid transparent",
                  padding: "7px 9px", cursor: "pointer", textAlign: "left",
                  borderRadius: "8px", color: active ? "#fff" : "#a8adbd",
                  fontSize: "12.5px", fontFamily: UI_FONT,
                  fontWeight: active ? 600 : 400, transition: "background .1s",
                }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = "#181a21"; }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = "none"; }}
                >
                  <span style={{
                    width: "7px", height: "7px", borderRadius: "50%", flexShrink: 0,
                    background: t.color,
                    boxShadow: active ? `0 0 0 3px ${t.color}26` : "none",
                  }} />
                  {t.label}
                </button>
              );
            })}
          </div>

          <div style={{ flex: 1 }} />

          {/* bottom nav: chat / precedents / kb / export / admin */}
          <div style={{ padding: "10px", borderTop: "1px solid #1e212a" }}>
            {[
              ...(currentUser && !currentUser.is_admin
                ? [{ id: "dashboard", label: "My Dashboard", icon: IconGrid, onClick: () => { setMainView("dashboard"); fetchMeStats(); } }]
                : []),
              { id: "chat",       label: "Chat",                          icon: IconFile,    onClick: () => setMainView("chat") },
              { id: "precedents", label: "Case Library",                  icon: IconBook,    onClick: () => setMainView("precedents") },
              { id: "kb",         label: `Knowledge Base · ${kbCount}`,   icon: IconDatabase, onClick: () => { setMainView("kb"); fetchKbEntries(); } },
            ].map(item => {
              const active = mainView === item.id;
              const Icon = item.icon;
              return (
                <button key={item.id} onClick={item.onClick} style={{
                  display: "flex", alignItems: "center", gap: "10px", width: "100%",
                  background: active ? "#1d2029" : "none",
                  border: "none", borderLeft: active ? "3px solid #3b82f6" : "3px solid transparent",
                  color: active ? "#fff" : "#a8adbd", padding: "8px 9px",
                  borderRadius: "8px", cursor: "pointer", fontSize: "12.5px", fontFamily: UI_FONT,
                  fontWeight: active ? 600 : 400, marginBottom: "2px", transition: "background .1s",
                }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = "#181a21"; }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = "none"; }}
                ><Icon width={15} height={15} />{item.label}</button>
              );
            })}

            <button onClick={exportChat} style={{
              display: "flex", alignItems: "center", gap: "10px", width: "100%",
              background: "none", border: "none", borderLeft: "3px solid transparent",
              color: "#a8adbd", padding: "8px 9px",
              borderRadius: "8px", cursor: "pointer", fontSize: "12.5px", fontFamily: UI_FONT,
              marginBottom: currentUser?.is_admin ? "2px" : 0, transition: "background .1s",
            }}
              onMouseEnter={e => e.currentTarget.style.background = "#181a21"}
              onMouseLeave={e => e.currentTarget.style.background = "none"}
            ><IconDownload width={15} height={15} />Export Chat</button>

            {currentUser?.is_admin && (
              <button onClick={() => setMainView("admin")} style={{
                display: "flex", alignItems: "center", gap: "10px", width: "100%",
                background: mainView === "admin" ? "#2a1f3d" : "none",
                border: "none", borderLeft: mainView === "admin" ? "3px solid #a78bfa" : "3px solid transparent",
                color: mainView === "admin" ? "#e9d5ff" : "#a8adbd", padding: "8px 9px",
                borderRadius: "8px", cursor: "pointer", fontSize: "12.5px", fontFamily: UI_FONT,
                fontWeight: mainView === "admin" ? 600 : 400, transition: "background .1s",
              }}
                onMouseEnter={e => { if (mainView !== "admin") e.currentTarget.style.background = "#181a21"; }}
                onMouseLeave={e => { if (mainView !== "admin") e.currentTarget.style.background = "none"; }}
              ><IconShield width={15} height={15} />Admin Dashboard</button>
            )}

            {currentUser ? (
              <div style={{
                display: "flex", alignItems: "center", gap: "9px",
                marginTop: "10px", padding: "10px 9px 2px", borderTop: "1px solid #1e212a",
              }}>
                <div style={{
                  width: "26px", height: "26px",
                  background: currentUser.is_admin ? "linear-gradient(135deg,#a78bfa,#7c3aed)" : "linear-gradient(135deg,#60a5fa,#1d4ed8)",
                  borderRadius: "50%",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#fff", fontSize: "10px", fontWeight: 700, flexShrink: 0,
                }}>{(currentUser.username || "?").slice(0, 2).toUpperCase()}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: "12px", color: "#dcdfe8", fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {currentUser.username}
                  </div>
                  {currentUser.is_admin && (
                    <div style={{ fontSize: "9.5px", color: "#a78bfa", fontWeight: 600, letterSpacing: ".03em" }}>ADMIN</div>
                  )}
                </div>
                <button onClick={logout} title="Log out" style={{
                  background: "none", border: "none", color: "#5b6072", cursor: "pointer",
                  padding: "6px", borderRadius: "7px", display: "flex", flexShrink: 0, transition: "background .12s, color .12s",
                }}
                  onMouseEnter={e => { e.currentTarget.style.background = "#1c1f28"; e.currentTarget.style.color = "#ececec"; }}
                  onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#5b6072"; }}
                ><IconLogout width={14} height={14} /></button>
              </div>
            ) : (
              <button onClick={() => setShowAuthModal(true)} style={{
                display: "flex", alignItems: "center", gap: "10px", width: "100%",
                marginTop: "10px", padding: "10px 9px", borderTop: "1px solid #1e212a",
                background: "none", border: "none", borderTopWidth: "1px", borderTopStyle: "solid", borderTopColor: "#1e212a",
                color: "#a8adbd", cursor: "pointer", fontSize: "12.5px", fontFamily: UI_FONT, fontWeight: 500,
              }}
                onMouseEnter={e => e.currentTarget.style.color = "#fff"}
                onMouseLeave={e => e.currentTarget.style.color = "#a8adbd"}
              ><IconUser width={15} height={15} />Sign In</button>
            )}
          </div>
        </div>
      </div>

      {showAuthModal && (
        <UserAuth
          setOpenAuth={setShowAuthModal}
          setAuthToken={setAuthToken}
          setNotification={setNotification}
          setOpenSurvey={() => {}}
          getHistory={() => {}}
        />
      )}

      {notification && (
        <div style={{
          position: "fixed", bottom: "20px", left: "50%", transform: "translateX(-50%)",
          background: "#111827", color: "#fff", padding: "10px 18px", borderRadius: "8px",
          fontSize: "13px", fontFamily: UI_FONT, boxShadow: "0 4px 16px rgba(0,0,0,.2)",
          // above the auth modal's overlay (z-index 9999 in UserAuth.css) — a login
          // error is often shown *while that modal is open*, so this must render on
          // top of it or the toast is invisible even though it's firing correctly.
          zIndex: 10000,
        }}>{notification}</div>
      )}

      {/* ── MAIN COLUMN ────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, background: "#fff" }}>

        {/* header */}
        <div style={{
          height: "52px", flexShrink: 0, display: "flex", alignItems: "center",
          gap: "10px", padding: "0 14px", borderBottom: "1px solid #f0f0f0",
        }}>
          {!sidebarOpen && (
            <button onClick={() => setSidebarOpen(true)} title="Open sidebar" style={{
              background: "none", border: "none", color: "#6b7280", cursor: "pointer",
              padding: "5px", borderRadius: "6px", display: "flex",
            }}><IconPanel width={18} height={18} /></button>
          )}
          <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "13.5px", fontWeight: 600, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {mainView === "chat" ? (isAllScope ? `All Documents (${docs.length})` : activeDoc ? activeDoc.name : "SynergeReader Legal AI")
                : mainView === "precedents" ? "Case Library"
                : mainView === "admin" ? "Admin Dashboard"
                : mainView === "dashboard" ? "My Dashboard"
                : "Knowledge Base"}
            </span>
            {mainView === "chat" && (activeDoc || isAllScope) && (
              <Badge color={task?.color || "#0891b2"}>{modelName}</Badge>
            )}
          </div>
          {mainView === "chat" && activeDoc && (
            <button onClick={() => setSourceOpen(o => !o)} title={sourceOpen ? "Hide source panel" : "Show source panel"} style={{
              display: "flex", alignItems: "center", gap: "6px",
              background: sourceOpen ? "#eff6ff" : "#f8fafc",
              border: `1px solid ${sourceOpen ? "#bfdbfe" : "#e5e7eb"}`,
              color: sourceOpen ? "#1d4ed8" : "#4b5563",
              borderRadius: "8px", padding: "6px 10px", cursor: "pointer",
              fontSize: "12px", fontFamily: UI_FONT, fontWeight: 500,
            }}><IconPanel width={14} height={14} />{sourceOpen ? "Hide Source" : "Show Source"}</button>
          )}

          {/* ── account menu — top-right of the (light) chat header, same
              click-to-open dropdown pattern GitHub/Slack/etc. use ── */}
          {currentUser && (
            <div ref={userMenuRef} style={{ position: "relative", flexShrink: 0 }}>
              <button
                onClick={() => setUserMenuOpen(o => !o)}
                title={currentUser.username}
                style={{
                  display: "flex", alignItems: "center", gap: "5px",
                  background: userMenuOpen ? "#f3f4f6" : "transparent",
                  border: "none", borderRadius: "20px", padding: "3px 7px 3px 3px",
                  cursor: "pointer", transition: "background .12s",
                }}
                onMouseEnter={e => { if (!userMenuOpen) e.currentTarget.style.background = "#f8fafc"; }}
                onMouseLeave={e => { if (!userMenuOpen) e.currentTarget.style.background = "transparent"; }}
              >
                <div style={{
                  width: "26px", height: "26px", borderRadius: "50%",
                  background: currentUser.is_admin ? "linear-gradient(135deg,#a78bfa,#7c3aed)" : "linear-gradient(135deg,#60a5fa,#1d4ed8)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#fff", fontSize: "10px", fontWeight: 700, flexShrink: 0,
                }}>{(currentUser.username || "?").slice(0, 2).toUpperCase()}</div>
                <IconChevronDown width={13} height={13} style={{ color: "#9ca3af" }} />
              </button>

              {userMenuOpen && (
                <div style={{
                  position: "absolute", top: "calc(100% + 8px)", right: 0,
                  width: "236px", background: "#fff", border: "1px solid #e5e7eb",
                  borderRadius: "12px", boxShadow: "0 10px 30px rgba(15,23,42,.14), 0 2px 8px rgba(15,23,42,.06)",
                  zIndex: 50, overflow: "hidden", fontFamily: UI_FONT,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "14px" }}>
                    <div style={{
                      width: "36px", height: "36px", borderRadius: "50%",
                      background: currentUser.is_admin ? "linear-gradient(135deg,#a78bfa,#7c3aed)" : "linear-gradient(135deg,#60a5fa,#1d4ed8)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: "#fff", fontSize: "13px", fontWeight: 700, flexShrink: 0,
                    }}>{(currentUser.username || "?").slice(0, 2).toUpperCase()}</div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span style={{ fontSize: "13px", fontWeight: 700, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {currentUser.username}
                        </span>
                        {currentUser.is_admin && (
                          <span style={{
                            fontSize: "9px", fontWeight: 700, color: "#7c3aed", background: "#f3e8ff",
                            padding: "1.5px 6px", borderRadius: "20px", letterSpacing: ".03em", flexShrink: 0,
                          }}>ADMIN</span>
                        )}
                      </div>
                      {currentUser.email && (
                        <div style={{ fontSize: "11.5px", color: "#6b7280", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {currentUser.email}
                        </div>
                      )}
                    </div>
                  </div>

                  <div style={{ height: "1px", background: "#f0f0f0" }} />

                  <div style={{ padding: "6px" }}>
                    {/* "My Dashboard" is a personal-activity view for regular
                        members only — admins get Admin Dashboard instead,
                        same split the sidebar nav already makes. Profile
                        (below) is the same personal view for everyone —
                        Admin Dashboard itself stays reachable from the
                        sidebar, not duplicated in this menu. */}
                    <button
                      onClick={() => { setMainView("dashboard"); fetchMeStats(); setUserMenuOpen(false); }}
                      style={{
                        display: "flex", alignItems: "center", gap: "9px", width: "100%",
                        background: "none", border: "none", padding: "8px 8px", borderRadius: "8px",
                        color: "#374151", fontSize: "12.5px", fontFamily: UI_FONT, cursor: "pointer", textAlign: "left",
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = "#f8fafc"}
                      onMouseLeave={e => e.currentTarget.style.background = "none"}
                    ><IconUser width={15} height={15} />Profile</button>

                    <button
                      onClick={() => { exportChat(); setUserMenuOpen(false); }}
                      style={{
                        display: "flex", alignItems: "center", gap: "9px", width: "100%",
                        background: "none", border: "none", padding: "8px 8px", borderRadius: "8px",
                        color: "#374151", fontSize: "12.5px", fontFamily: UI_FONT, cursor: "pointer", textAlign: "left",
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = "#f8fafc"}
                      onMouseLeave={e => e.currentTarget.style.background = "none"}
                    ><IconDownload width={15} height={15} />Export Chat</button>
                  </div>

                  <div style={{ height: "1px", background: "#f0f0f0" }} />

                  <div style={{ padding: "6px" }}>
                    <button
                      onClick={() => { setUserMenuOpen(false); logout(); }}
                      style={{
                        display: "flex", alignItems: "center", gap: "9px", width: "100%",
                        background: "none", border: "none", padding: "8px 8px", borderRadius: "8px",
                        color: "#dc2626", fontSize: "12.5px", fontFamily: UI_FONT, cursor: "pointer", textAlign: "left", fontWeight: 500,
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = "#fef2f2"}
                      onMouseLeave={e => e.currentTarget.style.background = "none"}
                    ><IconLogout width={15} height={15} />Log out</button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── CHAT VIEW ── */}
        {mainView === "chat" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>

            {!hasDocs ? (
              // welcome / upload-first hero — the "chat" IS the upload surface
              <div style={{
                flex: 1, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", padding: "24px",
              }}>
                <div style={{
                  width: "48px", height: "48px", marginBottom: "18px",
                  background: "linear-gradient(135deg,#3b82f6,#1d4ed8)",
                  borderRadius: "12px", display: "flex", alignItems: "center",
                  justifyContent: "center", color: "#fff",
                }}><IconScale width={26} height={26} /></div>
                <div style={{ fontSize: "22px", fontWeight: 700, color: "#111827", marginBottom: "6px" }}>
                  Upload a legal document to begin
                </div>
                <div style={{ fontSize: "13px", color: "#6b7280", marginBottom: "28px", textAlign: "center", lineHeight: "1.6" }}>
                  PDF, DOCX or TXT — up to 20 MB. Attach a file below or drag it anywhere on this page,<br />
                  then ask questions the way you would in any AI chat.
                </div>
                <div style={{ width: "100%", maxWidth: "640px" }}>
                  <Composer
                    input={input}
                    setInput={setInput}
                    onSend={() => sendMessage(input)}
                    onAttach={() => fileInputRef.current?.click()}
                    uploading={uploading}
                    disabled={true}
                    placeholder="Upload a document first…"
                  />
                </div>
                {uploadErr && (
                  <div style={{
                    marginTop: "14px", background: "#fee2e2", border: "1px solid #fca5a5",
                    borderRadius: "8px", padding: "8px 14px", color: "#991b1b", fontSize: "12px",
                  }}>{uploadErr}</div>
                )}
              </div>
            ) : (
              <>
                {/* combined-scope banner — makes it explicit which documents are in play */}
                {isAllScope && (
                  <div style={{
                    flexShrink: 0, padding: "8px 20px", borderBottom: "1px solid #f0f0f0",
                    display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap",
                    background: "#f8fafc",
                  }}>
                    <span style={{ fontSize: "11px", color: "#64748b", fontWeight: 600 }}>Asking across:</span>
                    {docs.map(d => (
                      <button key={d.id} onClick={() => { setActiveDocId(d.id); setHlPage(null); }} style={{
                        background: "#fff", border: "1px solid #e2e8f0", borderRadius: "12px",
                        padding: "3px 10px", fontSize: "11px", color: "#374151", cursor: "pointer",
                        fontFamily: UI_FONT,
                      }}>{d.name}</button>
                    ))}
                  </div>
                )}

                {activeTask !== "research" ? (
                  <ToolPanel
                    task={activeTask}
                    taskMeta={task}
                    hasDocs={hasDocs}
                    activeDocId={activeDocId}
                    toolLoading={toolLoading}
                    toolError={toolError}
                    argumentEntries={argumentByDoc[activeDocId] || []}
                    argumentClaim={argumentClaim}
                    setArgumentClaim={setArgumentClaim}
                    onRunArgument={runArgumentTool}
                    riskResult={riskByDoc[activeDocId]}
                    onRunRisk={runRiskTool}
                    clauseEntries={clauseByDoc[activeDocId] || []}
                    customClauseInput={customClauseInput}
                    setCustomClauseInput={setCustomClauseInput}
                    onRunClause={runClauseTool}
                    summaryResult={summaryByDoc[activeDocId]}
                    summaryLength={summaryLength}
                    setSummaryLength={setSummaryLength}
                    onRunSummary={runSummaryTool}
                    precedentsResult={precedentsToolByDoc[activeDocId]}
                    onRunPrecedents={runPrecedentsTool}
                    onCitation={handleCitation}
                  />
                ) : (
                <>
                {/* messages */}
                <div ref={chatScrollRef} style={{ flex: 1, overflow: "auto" }}>
                  <div style={{ maxWidth: "760px", margin: "0 auto", padding: "24px 20px 8px" }}>
                    {messages.length === 0 && (
                      <div style={{ color: "#9ca3af", fontSize: "13px", textAlign: "center", marginTop: "40px" }}>
                        {isAllScope
                          ? `Ask a question across all ${docs.length} documents to begin.`
                          : `Ask a question about "${activeDoc?.name}" to begin.`}
                      </div>
                    )}

                    {messages.map(msg => {
                      const isUser = msg.role === "user";
                      if (isUser) {
                        return (
                          <div key={msg.id} style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", marginBottom: "18px" }}>
                            {msg.scope && (
                              <div style={{ fontSize: "10px", color: "#9ca3af", marginBottom: "3px", paddingRight: "4px" }}>
                                {msg.scope.type === "all" ? `📎 All Documents (${msg.scope.count})` : `📎 ${msg.scope.name}`}
                              </div>
                            )}
                            <div style={{
                              maxWidth: "75%", background: "#f4f4f5",
                              borderRadius: "20px", padding: "10px 16px",
                              fontSize: "14.5px", lineHeight: "1.6", color: "#0d0d0d",
                              whiteSpace: "pre-wrap",
                            }}>{msg.text}</div>
                          </div>
                        );
                      }
                      return (
                        <div key={msg.id} style={{ display: "flex", gap: "12px", marginBottom: "22px" }}>
                          <div style={{
                            width: "28px", height: "28px", borderRadius: "8px", flexShrink: 0,
                            background: "linear-gradient(135deg,#3b82f6,#1d4ed8)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            color: "#fff", marginTop: "2px",
                          }}><IconScale width={15} height={15} /></div>
                          <div style={{ minWidth: 0, flex: 1 }}>
                            {msg.model && (
                              <div style={{
                                fontSize: "10.5px", fontFamily: "'Courier New',monospace",
                                color: task?.color || "#0891b2", fontWeight: 700,
                                marginBottom: "4px", letterSpacing: ".04em", textTransform: "uppercase",
                              }}>{MODEL_LABEL[msg.model] || msg.model}</div>
                            )}
                            <div style={{
                              fontSize: "14.5px", color: "#1e293b", lineHeight: "1.7",
                              whiteSpace: "pre-wrap",
                            }}>
                              {msg.text}
                              {msg.streaming && <span style={{ opacity: .5, animation: "blink 1s infinite" }}>▊</span>}
                            </div>
                            {msg.citations?.length > 0 && (
                              <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "5px" }}>
                                {msg.citations.map((c, i) => (
                                  <CitationChip key={i} page={c.page} label="" onClick={handleCitation} />
                                ))}
                              </div>
                            )}
                            {!msg.streaming && msg.entryId && (
                              <div style={{ display: "flex", alignItems: "center", gap: "4px", marginTop: "8px" }}>
                                <button
                                  onClick={() => submitRating(msg.id, msg.entryId, 5)}
                                  title="Good response"
                                  style={{
                                    background: msg.userRating === 5 ? "#dcfce7" : "none",
                                    border: "1px solid", borderColor: msg.userRating === 5 ? "#86efac" : "transparent",
                                    color: msg.userRating === 5 ? "#16a34a" : "#9ca3af",
                                    cursor: "pointer", padding: "4px", borderRadius: "6px", display: "flex",
                                  }}
                                  onMouseEnter={e => { if (msg.userRating !== 5) e.currentTarget.style.background = "#f3f4f6"; }}
                                  onMouseLeave={e => { if (msg.userRating !== 5) e.currentTarget.style.background = "transparent"; }}
                                ><IconThumbUp width={13} height={13} /></button>
                                <button
                                  onClick={() => submitRating(msg.id, msg.entryId, 1)}
                                  title="Poor response"
                                  style={{
                                    background: msg.userRating === 1 ? "#fee2e2" : "none",
                                    border: "1px solid", borderColor: msg.userRating === 1 ? "#fca5a5" : "transparent",
                                    color: msg.userRating === 1 ? "#dc2626" : "#9ca3af",
                                    cursor: "pointer", padding: "4px", borderRadius: "6px", display: "flex",
                                  }}
                                  onMouseEnter={e => { if (msg.userRating !== 1) e.currentTarget.style.background = "#f3f4f6"; }}
                                  onMouseLeave={e => { if (msg.userRating !== 1) e.currentTarget.style.background = "transparent"; }}
                                ><IconThumbDown width={13} height={13} /></button>
                                {msg.userRating && (
                                  <span style={{ fontSize: "10.5px", color: "#9ca3af" }}>Thanks for the feedback</span>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}

                    {typing && (
                      <div style={{ display: "flex", gap: "12px", marginBottom: "18px" }}>
                        <div style={{
                          width: "28px", height: "28px", borderRadius: "8px", flexShrink: 0,
                          background: "linear-gradient(135deg,#3b82f6,#1d4ed8)",
                          display: "flex", alignItems: "center", justifyContent: "center", color: "#fff",
                        }}><IconScale width={15} height={15} /></div>
                        <DotsLoader />
                      </div>
                    )}
                  </div>
                </div>

                {/* composer + suggestions */}
                <div style={{ flexShrink: 0, padding: "8px 20px 18px" }}>
                  <div style={{ maxWidth: "760px", margin: "0 auto" }}>
                    {suggestions.length > 0 && messages.length < 3 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "10px" }}>
                        {suggestions.map((q, i) => (
                          <button key={i} onClick={() => sendMessage(q)} style={{
                            background: "#f8fafc", border: "1px solid #e5e7eb",
                            borderRadius: "16px", padding: "6px 12px",
                            textAlign: "left", fontSize: "12px", color: "#374151",
                            cursor: "pointer", fontFamily: UI_FONT,
                          }}
                            onMouseEnter={e => e.currentTarget.style.background = "#eff6ff"}
                            onMouseLeave={e => e.currentTarget.style.background = "#f8fafc"}
                          >{q}</button>
                        ))}
                      </div>
                    )}
                    {selectedContext && (
                      <div style={{
                        display: "flex", alignItems: "flex-start", gap: "8px",
                        background: "#eff6ff", border: "1px solid #bfdbfe",
                        borderRadius: "10px", padding: "8px 10px", marginBottom: "8px",
                      }}>
                        <IconScale width={13} height={13} style={{ color: "#2563eb", flexShrink: 0, marginTop: "2px" }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#1d4ed8", textTransform: "uppercase", letterSpacing: ".04em", marginBottom: "2px" }}>
                            Asking about this selection
                          </div>
                          <div style={{
                            fontSize: "12px", color: "#1e3a8a", lineHeight: "1.4",
                            display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden",
                          }}>{selectedContext.text}</div>
                        </div>
                        <button
                          onClick={() => setSelectedContext(null)}
                          title="Clear selection — ask about the whole document instead"
                          style={{
                            background: "none", border: "none", color: "#60a5fa", cursor: "pointer",
                            padding: "2px", display: "flex", flexShrink: 0,
                          }}
                        ><IconX width={14} height={14} /></button>
                      </div>
                    )}
                    <Composer
                      input={input}
                      setInput={setInput}
                      onSend={() => sendMessage(input)}
                      onAttach={() => fileInputRef.current?.click()}
                      uploading={uploading}
                      disabled={(!activeDoc && !isAllScope) || typing}
                      placeholder={selectedContext ? "Ask a question about the selected text…" : isAllScope ? `Ask across all ${docs.length} documents…` : activeDoc ? "Ask a legal question…" : "Select a document…"}
                      autoFocus={!!selectedContext}
                    />
                    {uploadErr && (
                      <div style={{
                        marginTop: "8px", background: "#fee2e2", border: "1px solid #fca5a5",
                        borderRadius: "8px", padding: "6px 12px", color: "#991b1b", fontSize: "11.5px",
                      }}>{uploadErr}</div>
                    )}
                    <div style={{ textAlign: "center", fontSize: "10.5px", color: "#9ca3af", marginTop: "8px" }}>
                      All data stays on your server · {modelName} active
                    </div>
                  </div>
                </div>
                </>
                )}
              </>
            )}
          </div>
        )}

        {/* ── CASE LIBRARY / PRECEDENTS VIEW ── */}
        {mainView === "precedents" && (
          <div style={{ flex: 1, overflow: "auto", padding: "24px 20px" }}>
            <div style={{ maxWidth: "760px", margin: "0 auto" }}>
              <div style={{ fontSize: "18px", fontWeight: 700, color: "#111827", marginBottom: "4px" }}>Related Precedents</div>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "18px" }}>Sourced live from CourtListener based on the active document.</div>

              {precLoading && (
                <div style={{ fontSize: "12px", color: "#94a3b8" }}>Searching CourtListener…</div>
              )}

              {!precLoading && precedents.length === 0 && (
                <div style={{
                  padding: "16px", background: "#f8fafc", border: "1px solid #e2e8f0",
                  borderRadius: "10px", fontSize: "12.5px", color: "#64748b", lineHeight: "1.6",
                }}>
                  {activeDoc
                    ? "No precedents found. CourtListener API may be unavailable or no matching cases found."
                    : "Upload a document to search for related precedents automatically."}
                </div>
              )}

              {precedents.map((p, i) => {
                const col = i === 0 ? "#16a34a" : i === 1 ? "#d97706" : "#64748b";
                return (
                  <div key={i} style={{
                    padding: "14px", background: "#f8fafc",
                    border: "1px solid #e2e8f0", borderRadius: "10px",
                    marginBottom: "8px", cursor: p.url ? "pointer" : "default",
                  }}
                    onClick={() => p.url && window.open(p.url, "_blank")}
                    onMouseEnter={e => e.currentTarget.style.background = "#eff6ff"}
                    onMouseLeave={e => e.currentTarget.style.background = "#f8fafc"}
                  >
                    <div style={{ fontSize: "13.5px", fontWeight: 600, color: "#1d4ed8", marginBottom: "3px" }}>{p.name}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: p.snippet ? "6px" : "8px" }}>
                      {p.court}{p.date ? ` · ${p.date}` : ""}
                      {p.url && <span style={{ color: "#3b82f6", marginLeft: "6px" }}>↗ View</span>}
                    </div>
                    {p.snippet && (
                      <div style={{ fontSize: "12px", color: "#475569", lineHeight: "1.5", marginBottom: "8px", fontStyle: "italic" }}>
                        {p.snippet.slice(0, 160)}…
                      </div>
                    )}
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <div style={{ flex: 1, height: "4px", background: "#e5e7eb", borderRadius: "2px", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${p.score !== null ? p.score + "%" : "0%"}`, background: col, borderRadius: "2px" }} />
                      </div>
                      <span style={{ fontSize: "11px", fontWeight: 700, color: col, fontFamily: "'Courier New',monospace" }}>
                        {p.score !== null ? p.score + "%" : "N/A"}
                      </span>
                    </div>
                  </div>
                );
              })}

              {!precLoading && activeDoc && (
                <button
                  onClick={() => searchPrecedents(activeDoc.text?.slice(0, 300) || "", activeDoc.name)}
                  style={{
                    marginTop: "6px", background: "#f8fafc", border: "1px solid #e2e8f0",
                    borderRadius: "8px", padding: "8px 14px", cursor: "pointer",
                    fontSize: "12px", color: "#374151", fontFamily: UI_FONT,
                  }}
                >↺ Search Again</button>
              )}
            </div>
          </div>
        )}

        {/* ── KNOWLEDGE BASE VIEW ── */}
        {mainView === "kb" && (() => {
          const categoryCounts = {
            all: kbEntries.length,
            document: kbEntries.filter(e => (e.source_type || "document") === "document").length,
            external_url: kbEntries.filter(e => e.source_type === "external_url").length,
            external_import: kbEntries.filter(e => e.source_type === "external_import").length,
            manual: kbEntries.filter(e => e.source_type === "manual").length,
          };
          const filtered = kbEntries.filter(e => {
            const inCategory = kbCategory === "all" || (e.source_type || "document") === kbCategory;
            const inSearch = !kbSearch ||
              e.question?.toLowerCase().includes(kbSearch.toLowerCase()) ||
              e.answer?.toLowerCase().includes(kbSearch.toLowerCase());
            return inCategory && inSearch;
          });
          const sorted = filtered.slice().sort((a, b) => {
            if (kbSort === "usage") return (b.usage_count || 0) - (a.usage_count || 0);
            if (kbSort === "recent") return new Date(b.created_at || 0) - new Date(a.created_at || 0);
            if (kbSort === "alpha") return (a.question || "").localeCompare(b.question || "");
            return 0;
          });
          const totalUsage = kbEntries.reduce((s, e) => s + (e.usage_count || 0), 0);
          const verifiedCount = kbEntries.filter(isKbVerified).length;
          const toggleExpand = id => setExpandedKb(prev => {
            const next = new Set(prev);
            next.has(id) ? next.delete(id) : next.add(id);
            return next;
          });
          const originOf = entry => {
            if (entry.source_type === "external_url") {
              let domain = entry.context_text;
              try { domain = new URL(entry.context_text).hostname; } catch { /* keep raw */ }
              return { tone: "auto", label: `Web · ${domain}` };
            }
            if (entry.source_type === "external_import") return { tone: "corrected", label: "Imported batch" };
            if (entry.source_type === "manual") return { tone: "manual", label: "Manual entry" };
            if ((entry.corrected_by || "").toLowerCase().includes("auto")) return { tone: "auto", label: "Auto-generated" };
            return { tone: "corrected", label: `Refined by ${entry.corrected_by}` };
          };
          const exportKbCsv = () => downloadCsv(
            `synergereader_kb_${new Date().toISOString().slice(0, 10)}.csv`,
            ["Question", "Answer", "Source Type", "Origin", "Verified", "Times Reused", "Created"],
            sorted.map(e => [e.question, e.answer, e.source_type, e.corrected_by, isKbVerified(e) ? "yes" : "no", e.usage_count, e.created_at])
          );
          const exportKbJsonl = () => {
            const set = kbVerifiedOnly ? sorted.filter(isKbVerified) : sorted;
            downloadJsonl(
              `synergereader_kb_training_${new Date().toISOString().slice(0, 10)}.jsonl`,
              set.map(e => ({
                messages: [{ role: "user", content: e.question }, { role: "assistant", content: e.answer }],
                source_type: e.source_type,
                verified: isKbVerified(e),
                usage_count: e.usage_count,
                created_at: e.created_at,
              }))
            );
          };

          const CATEGORY_CHIPS = [
            { id: "all", label: "All Sources" },
            { id: "document", label: "Document Q&A" },
            { id: "external_url", label: "External Sources" },
            { id: "external_import", label: "Imported Batches" },
            { id: "manual", label: "Manual Entries" },
          ];

          const inputStyle = {
            width: "100%", border: "1px solid #d1d5db", borderRadius: "9px", padding: "9px 11px",
            fontSize: "13px", fontFamily: UI_FONT, color: "#1e293b", outline: "none", boxSizing: "border-box",
          };
          const primaryBtnStyle = (busy, tint) => ({
            display: "flex", alignItems: "center", gap: "6px", background: busy ? "#e5e7eb" : tint,
            color: "#fff", border: "none", borderRadius: "9px", padding: "9px 16px", fontSize: "12.5px",
            fontWeight: 700, cursor: busy ? "default" : "pointer", fontFamily: UI_FONT, whiteSpace: "nowrap",
          });

          const KbActionCard = ({ id, icon: Icon, title, desc, gradient }) => (
            <button
              onClick={() => setKbTool(kbTool === id ? null : id)}
              style={{
                textAlign: "left", border: "none", borderRadius: "16px", padding: "18px",
                background: gradient, color: "#fff", cursor: "pointer", minHeight: "126px",
                display: "flex", flexDirection: "column", justifyContent: "space-between",
                boxShadow: kbTool === id ? "0 0 0 3px rgba(17,24,39,.18), 0 10px 22px rgba(0,0,0,.16)" : "0 3px 10px rgba(0,0,0,.08)",
                transition: "transform .15s, box-shadow .15s", fontFamily: UI_FONT,
              }}
              onMouseEnter={e => e.currentTarget.style.transform = "translateY(-2px)"}
              onMouseLeave={e => e.currentTarget.style.transform = "translateY(0)"}
            >
              <div style={{ width: "30px", height: "30px", borderRadius: "9px", background: "rgba(255,255,255,.22)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon width={16} height={16} />
              </div>
              <div>
                <div style={{ fontSize: "14px", fontWeight: 700, marginBottom: "4px" }}>{title}</div>
                <div style={{ fontSize: "11.5px", opacity: .88, lineHeight: "1.45" }}>{desc}</div>
              </div>
            </button>
          );

          const isKbAdmin = !!currentUser?.is_admin;

          return (
            <div style={{ flex: 1, overflow: "auto", padding: "28px 24px 40px" }}>
              <div style={{ maxWidth: "1400px", margin: "0 auto", display: "flex", gap: "20px", alignItems: "flex-start" }}>
              <div style={{ flex: "1 1 0", minWidth: 0 }}>

                {/* hero header */}
                <div style={{ marginBottom: "22px" }}>
                  <div style={{ fontSize: "24px", fontWeight: 700, color: "#111827", marginBottom: "6px", letterSpacing: "-.01em" }}>
                    SynergeReader Knowledge Base
                  </div>
                  <div style={{ fontSize: "13px", color: "#6b7280", lineHeight: "1.6", maxWidth: "620px" }}>
                    {isKbAdmin
                      ? "A growing, reusable set of verified question-answer pairs — built from your documents, your team's conversations, and any external source you bring in. It grounds every future answer, and it's structured to export directly into a model fine-tuning pipeline."
                      : "A growing, reusable set of verified question-answer pairs — built from your firm's documents, conversations, and trusted external sources. Every answer you get in chat draws on this shared knowledge."}
                  </div>
                </div>

                {/* get started — action cards (admin only; regular users get a read-only view) */}
                {isKbAdmin && (<>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", marginBottom: kbTool ? "14px" : "24px" }}>
                  <KbActionCard id="url" icon={IconGlobe} title="Add External Source"
                    desc="Import a webpage or article by URL." gradient="linear-gradient(135deg,#2563eb,#1d4ed8)" />
                  <KbActionCard id="manual" icon={IconEdit} title="Add Manual Entry"
                    desc="Write a verified Q&A pair yourself." gradient="linear-gradient(135deg,#7c3aed,#6d28d9)" />
                  <KbActionCard id="import" icon={IconUpload} title="Import Batch"
                    desc="Upload a JSON or CSV of Q&A pairs." gradient="linear-gradient(135deg,#059669,#047857)" />
                  <KbActionCard id="export" icon={IconDownload} title="Export for Training"
                    desc="Download as a fine-tuning dataset." gradient="linear-gradient(135deg,#d97706,#b45309)" />
                </div>

                {/* inline tool panels */}
                {kbTool === "url" && (
                  <div style={{ ...toolCardStyle, marginBottom: "24px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <IconGlobe width={16} height={16} color="#2563eb" />
                        <strong style={{ fontSize: "13px", color: "#111827" }}>Add External Source</strong>
                      </div>
                      <button onClick={() => setKbTool(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", display: "flex" }}><IconX width={16} height={16} /></button>
                    </div>
                    <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px", lineHeight: "1.5" }}>
                      Paste a URL to a webpage, statute, or reference article. SynergeReader reads it and generates Q&A pairs from it, the same way it does for an uploaded document.
                    </div>
                    <div style={{ display: "flex", gap: "8px" }}>
                      <input value={kbUrlInput} onChange={e => setKbUrlInput(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter") importKbFromUrl(); }}
                        placeholder="https://example.com/article" style={inputStyle} />
                      <button onClick={importKbFromUrl} disabled={!kbUrlInput.trim() || kbUrlBusy} style={primaryBtnStyle(!kbUrlInput.trim() || kbUrlBusy, "#2563eb")}>
                        {kbUrlBusy ? "Reading…" : "Import"}
                      </button>
                    </div>
                    {kbUrlMsg && (
                      <div style={{ marginTop: "10px", fontSize: "12px", color: kbUrlMsg.ok ? "#16a34a" : "#dc2626" }}>{kbUrlMsg.text}</div>
                    )}
                  </div>
                )}

                {kbTool === "manual" && (
                  <div style={{ ...toolCardStyle, marginBottom: "24px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <IconEdit width={16} height={16} color="#7c3aed" />
                        <strong style={{ fontSize: "13px", color: "#111827" }}>Add Manual Entry</strong>
                      </div>
                      <button onClick={() => setKbTool(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", display: "flex" }}><IconX width={16} height={16} /></button>
                    </div>
                    <div style={{ display: "grid", gap: "10px", marginBottom: "12px" }}>
                      <div>
                        <label style={{ fontSize: "11px", fontWeight: 600, color: "#374151", marginBottom: "5px", display: "block" }}>Question</label>
                        <input value={kbManualQ} onChange={e => setKbManualQ(e.target.value)} placeholder="e.g. What is our standard notice period?" style={inputStyle} />
                      </div>
                      <div>
                        <label style={{ fontSize: "11px", fontWeight: 600, color: "#374151", marginBottom: "5px", display: "block" }}>Answer</label>
                        <textarea value={kbManualA} onChange={e => setKbManualA(e.target.value)} rows={3} placeholder="The verified answer…" style={{ ...inputStyle, resize: "vertical" }} />
                      </div>
                      <div>
                        <label style={{ fontSize: "11px", fontWeight: 600, color: "#374151", marginBottom: "5px", display: "block" }}>Source (optional)</label>
                        <input value={kbManualSrc} onChange={e => setKbManualSrc(e.target.value)} placeholder="e.g. Firm policy handbook §4.2" style={inputStyle} />
                      </div>
                    </div>
                    <button onClick={addManualKbEntry} disabled={!kbManualQ.trim() || !kbManualA.trim() || kbManualBusy}
                      style={primaryBtnStyle(!kbManualQ.trim() || !kbManualA.trim() || kbManualBusy, "#7c3aed")}>
                      {kbManualBusy ? "Adding…" : "Add Entry"}
                    </button>
                    {kbManualMsg && (
                      <div style={{ marginTop: "10px", fontSize: "12px", color: kbManualMsg.ok ? "#16a34a" : "#dc2626" }}>{kbManualMsg.text}</div>
                    )}
                  </div>
                )}

                {kbTool === "import" && (
                  <div style={{ ...toolCardStyle, marginBottom: "24px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <IconUpload width={16} height={16} color="#059669" />
                        <strong style={{ fontSize: "13px", color: "#111827" }}>Import Batch</strong>
                      </div>
                      <button onClick={() => setKbTool(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", display: "flex" }}><IconX width={16} height={16} /></button>
                    </div>
                    <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px", lineHeight: "1.5" }}>
                      Bring in Q&A pairs from another system: a JSON array of <code>{`{question, answer, source?}`}</code>, or a CSV with <code>question</code>, <code>answer</code>, and optional <code>source</code> columns.
                    </div>
                    <input ref={kbImportFileRef} type="file" accept=".json,.csv" style={{ display: "none" }}
                      onChange={e => e.target.files[0] && handleKbImportFile(e.target.files[0])} />
                    <button onClick={() => kbImportFileRef.current?.click()} disabled={kbImportBusy} style={primaryBtnStyle(kbImportBusy, "#059669")}>
                      {kbImportBusy ? "Importing…" : "Choose File"}
                    </button>
                    {kbImportMsg && (
                      <div style={{ marginTop: "10px", fontSize: "12px", color: kbImportMsg.ok ? "#16a34a" : "#dc2626" }}>{kbImportMsg.text}</div>
                    )}
                  </div>
                )}

                {kbTool === "export" && (
                  <div style={{ ...toolCardStyle, marginBottom: "24px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <IconDownload width={16} height={16} color="#d97706" />
                        <strong style={{ fontSize: "13px", color: "#111827" }}>Export for Training</strong>
                      </div>
                      <button onClick={() => setKbTool(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", display: "flex" }}><IconX width={16} height={16} /></button>
                    </div>
                    <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px", lineHeight: "1.5" }}>
                      Exports respect the category filter and search below. JSONL uses the standard chat fine-tuning
                      shape (<code>messages: [user, assistant]</code>) plus source metadata for later filtering.
                    </div>
                    <label style={{ display: "flex", alignItems: "center", gap: "7px", fontSize: "12px", color: "#374151", marginBottom: "12px", cursor: "pointer" }}>
                      <input type="checkbox" checked={kbVerifiedOnly} onChange={e => setKbVerifiedOnly(e.target.checked)} />
                      Verified entries only ({verifiedCount} of {kbEntries.length}) — excludes raw, never-reviewed model output
                    </label>
                    <div style={{ display: "flex", gap: "8px" }}>
                      <button onClick={exportKbJsonl} disabled={!sorted.length} style={primaryBtnStyle(!sorted.length, "#d97706")}>JSONL — Chat format</button>
                      <button onClick={exportKbCsv} disabled={!sorted.length} style={{
                        display: "flex", alignItems: "center", gap: "6px", background: "#fff", border: "1px solid #e2e8f0",
                        borderRadius: "9px", padding: "9px 16px", fontSize: "12.5px", fontWeight: 700, color: "#374151",
                        cursor: sorted.length ? "pointer" : "default", fontFamily: UI_FONT,
                      }}>CSV</button>
                    </div>
                  </div>
                )}
                </>)}

                {/* stat row */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px", marginBottom: "18px" }}>
                  {[
                    { label: "Entries",          value: kbCount,                              icon: IconDatabase, tint: "#059669" },
                    { label: "Times Reused",     value: totalUsage,                           icon: IconThumbUp,  tint: "#2563eb" },
                    { label: "Verified",         value: verifiedCount,                         icon: IconShield,   tint: "#7c3aed" },
                    { label: "External Sources", value: categoryCounts.external_url,           icon: IconGlobe,    tint: "#d97706" },
                  ].map(card => (
                    <div key={card.label} style={{
                      background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "14px 16px",
                      boxShadow: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)",
                      display: "flex", alignItems: "center", justifyContent: "space-between",
                    }}>
                      <div>
                        <div style={{ fontSize: "10px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", fontWeight: 700, marginBottom: "4px" }}>{card.label}</div>
                        <div style={{ fontSize: "20px", fontWeight: 700, color: "#111827" }}>{card.value}</div>
                      </div>
                      <div style={{
                        width: "30px", height: "30px", borderRadius: "9px", background: card.tint + "16", color: card.tint,
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                      }}><card.icon width={15} height={15} /></div>
                    </div>
                  ))}
                </div>

                {/* category chips */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "14px" }}>
                  {CATEGORY_CHIPS.map(chip => (
                    <button key={chip.id} onClick={() => setKbCategory(chip.id)} style={{
                      background: kbCategory === chip.id ? "#111827" : "#fff",
                      color: kbCategory === chip.id ? "#fff" : "#374151",
                      border: `1px solid ${kbCategory === chip.id ? "#111827" : "#e2e8f0"}`,
                      borderRadius: "999px", padding: "6px 13px", fontSize: "12px", fontWeight: 600,
                      cursor: "pointer", fontFamily: UI_FONT, display: "flex", alignItems: "center", gap: "6px",
                    }}>{chip.label}<span style={{ opacity: .6, fontWeight: 500 }}>{categoryCounts[chip.id]}</span></button>
                  ))}
                </div>

                {/* search + sort + refresh */}
                <div style={{ display: "flex", gap: "8px", marginBottom: "14px" }}>
                  <div style={{ position: "relative", flex: 1 }}>
                    <div style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "#9ca3af", display: "flex" }}>
                      <IconSearch width={14} height={14} />
                    </div>
                    <input
                      value={kbSearch}
                      onChange={e => setKbSearch(e.target.value)}
                      placeholder="Search questions & answers…"
                      style={{
                        width: "100%", border: "1px solid #d1d5db", borderRadius: "10px",
                        padding: "8px 12px 8px 32px", fontSize: "13px", fontFamily: UI_FONT,
                        color: "#1e293b", outline: "none", boxSizing: "border-box",
                      }}
                    />
                  </div>
                  <select value={kbSort} onChange={e => setKbSort(e.target.value)} style={{
                    border: "1px solid #d1d5db", borderRadius: "10px", padding: "0 10px",
                    fontSize: "12.5px", fontFamily: UI_FONT, color: "#374151", background: "#fff", cursor: "pointer",
                  }}>
                    <option value="usage">Most used</option>
                    <option value="recent">Newest</option>
                    <option value="alpha">A → Z</option>
                  </select>
                  <button onClick={fetchKbEntries} title="Refresh" style={{
                    background: "#fff", border: "1px solid #e2e8f0", borderRadius: "10px",
                    padding: "0 10px", cursor: "pointer", display: "flex", alignItems: "center", color: "#6b7280",
                  }}><IconRefresh width={13} height={13} /></button>
                </div>

                {kbLoading && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "12px" }}>
                    {[0, 1, 2, 3].map(i => (
                      <div key={i} style={{ height: "128px", borderRadius: "14px", background: "#f1f5f9", animation: "pulse 1.4s ease-in-out infinite" }} />
                    ))}
                  </div>
                )}

                {!kbLoading && sorted.length === 0 && (
                  <div style={{
                    padding: "40px 20px", background: "#fff", border: "1px solid #eef0f3",
                    borderRadius: "14px", textAlign: "center",
                  }}>
                    <div style={{ fontSize: "32px", marginBottom: "10px", opacity: .3 }}>🧠</div>
                    <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151", marginBottom: "4px" }}>
                      {kbSearch || kbCategory !== "all" ? "No entries match these filters." : "No entries yet"}
                    </div>
                    <div style={{ fontSize: "12px", color: "#9ca3af" }}>
                      {kbSearch || kbCategory !== "all" ? "Try clearing the search or category filter." : "Ask a question, add a manual entry, or import an external source above."}
                    </div>
                  </div>
                )}

                {!kbLoading && sorted.length > 0 && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "12px" }}>
                    {sorted.map(entry => {
                      const isOpen = expandedKb.has(entry.id);
                      const origin = originOf(entry);
                      return (
                        <div key={entry.id}
                          onClick={() => toggleExpand(entry.id)}
                          style={{
                            background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "16px",
                            boxShadow: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)",
                            cursor: "pointer", transition: "box-shadow .15s", display: "flex", flexDirection: "column",
                          }}
                          onMouseEnter={e => e.currentTarget.style.boxShadow = "0 4px 14px rgba(16,24,40,.09)"}
                          onMouseLeave={e => e.currentTarget.style.boxShadow = "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)"}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "9px", gap: "8px" }}>
                            <Pill tone={origin.tone}>{origin.label}</Pill>
                            <div style={{ display: "flex", alignItems: "center", gap: "6px", flexShrink: 0 }}>
                              {isKbVerified(entry) && (
                                <span title="Verified" style={{ color: "#16a34a", display: "flex" }}><IconCheckCircle width={13} height={13} /></span>
                              )}
                              {isKbAdmin && (
                                <button
                                  onClick={e => { e.stopPropagation(); deleteKbEntry(entry.id); }}
                                  title="Delete entry"
                                  style={{ background: "none", border: "none", cursor: "pointer", color: "#c3c2b7", padding: "2px", display: "flex" }}
                                  onMouseEnter={e => e.currentTarget.style.color = "#dc2626"}
                                  onMouseLeave={e => e.currentTarget.style.color = "#c3c2b7"}
                                ><IconTrash width={13} height={13} /></button>
                              )}
                            </div>
                          </div>

                          <div style={{ fontSize: "13px", fontWeight: 700, color: "#111827", lineHeight: "1.4", marginBottom: "6px", ...(isOpen ? {} : clampStyle(2)) }}>
                            {entry.question}
                          </div>
                          <div style={{ fontSize: "12.5px", color: "#4b5563", lineHeight: "1.55", ...(isOpen ? {} : clampStyle(3)) }}>
                            {entry.answer}
                          </div>

                          {isOpen && entry.context_text && (
                            <div style={{ marginTop: "10px", padding: "9px 11px", background: "#f8fafc", borderRadius: "9px" }}>
                              <div style={{ fontSize: "9.5px", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: "4px" }}>
                                {entry.source_type === "external_url" ? "Source URL" : "Source"}
                              </div>
                              <div style={{ fontSize: "11px", color: "#6b7280", lineHeight: "1.5", wordBreak: "break-word" }}>
                                {entry.context_text.slice(0, 320)}{entry.context_text.length > 320 ? "…" : ""}
                              </div>
                            </div>
                          )}

                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "12px", paddingTop: "10px", borderTop: "1px solid #f4f5f7" }}>
                            <span style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "10.5px", color: "#9ca3af", fontWeight: 600 }}>
                              <IconThumbUp width={11} height={11} />{entry.usage_count || 0}× reused
                            </span>
                            <span style={{ fontSize: "10.5px", color: "#c3c2b7" }}>{timeAgo(entry.created_at)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* right rail — real content that fills the width (coverage, breakdown,
                  most-reused), not the same cards stretched wider */}
              <div style={{ width: "300px", flexShrink: 0, display: "flex", flexDirection: "column", gap: "14px" }}>
                <ChartCard title="Verification Coverage" subtitle="Share of entries a human has verified">
                  <div style={{ display: "flex", justifyContent: "center", padding: "6px 0 2px" }}>
                    <RadialProgress value={verifiedCount} max={kbCount || 1} color="#7c3aed"
                      label="Verified" sublabel={`${verifiedCount} of ${kbCount}`} />
                  </div>
                </ChartCard>

                <ChartCard title="By Source">
                  <CategoricalBarList data={CATEGORY_CHIPS.filter(c => c.id !== "all").map(c => ({ label: c.label, value: categoryCounts[c.id] }))} />
                </ChartCard>

                <ChartCard title="Most Reused" subtitle="Answers the AI has drawn on most">
                  {kbEntries.filter(e => (e.usage_count || 0) > 0).length === 0 ? (
                    <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No entries reused yet.</div>
                  ) : (
                    <div>
                      {kbEntries.slice().sort((a, b) => (b.usage_count || 0) - (a.usage_count || 0)).slice(0, 5).map((e, i) => (
                        <div key={e.id} style={{
                          display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px",
                          padding: "8px 0", borderTop: i === 0 ? "none" : "1px solid #f4f5f7",
                        }}>
                          <div style={{ fontSize: "11.5px", color: "#374151", lineHeight: "1.4", ...clampStyle(2) }}>{e.question}</div>
                          <span style={{ fontSize: "10.5px", fontWeight: 700, color: "#2563eb", flexShrink: 0, whiteSpace: "nowrap" }}>{e.usage_count}×</span>
                        </div>
                      ))}
                    </div>
                  )}
                </ChartCard>
              </div>
              </div>
            </div>
          );
        })()}

        {/* ── PROFILE / MY DASHBOARD — personal, not system-wide: this user's own
             documents/chats/ratings, nothing about other users. Reachable from
             the sidebar (regular users, as "My Dashboard") and from the header
             account menu (everyone, including admins, as "Profile") — same
             view either way, since /me/stats is already scoped to the caller's
             own id regardless of admin status. Admins keep their separate,
             system-wide Admin Dashboard via the sidebar. ── */}
        {mainView === "dashboard" && (
          currentUser ? (
            <div style={{ flex: 1, overflow: "auto", padding: "26px 32px" }}>
              <div style={{ maxWidth: "1360px", margin: "0 auto" }}>

                {/* profile header card — name, email, avatar; a proper "who am
                    I signed in as" summary, not just a corner label */}
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px",
                  background: "#fff", border: "1px solid #eef0f3", borderRadius: "16px",
                  padding: "18px 20px", marginBottom: "18px", boxShadow: CARD_SHADOW,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "14px", minWidth: 0 }}>
                    <div style={{
                      width: "52px", height: "52px", borderRadius: "50%", flexShrink: 0,
                      background: currentUser.is_admin ? "linear-gradient(135deg,#a78bfa,#7c3aed)" : "linear-gradient(135deg,#60a5fa,#1d4ed8)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: "#fff", fontSize: "17px", fontWeight: 700,
                      boxShadow: "0 3px 10px rgba(37,99,235,.25)",
                    }}>{(currentUser.username || "?").slice(0, 2).toUpperCase()}</div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <div style={{ fontSize: "17px", fontWeight: 700, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {currentUser.username}
                        </div>
                        <span style={{
                          fontSize: "9.5px", fontWeight: 700, letterSpacing: ".04em", textTransform: "uppercase",
                          padding: "2px 8px", borderRadius: "20px", flexShrink: 0,
                          color: currentUser.is_admin ? "#7c3aed" : "#2563eb",
                          background: currentUser.is_admin ? "#f3e8ff" : "#eff6ff",
                        }}>{currentUser.is_admin ? "Admin" : "Member"}</span>
                      </div>
                      {currentUser.email && (
                        <div style={{ fontSize: "12.5px", color: "#6b7280", marginTop: "2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {currentUser.email}
                        </div>
                      )}
                    </div>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: "14px", flexShrink: 0 }}>
                    <div style={{ fontSize: "12px", color: "#9ca3af", whiteSpace: "nowrap" }}>
                      {new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" })}
                    </div>
                    {/* reuses the already-live, already-tested forgot-password
                        flow — a real security-relevant action, not just a
                        cosmetic profile field, so it earns its place here */}
                    <button
                      onClick={() => {
                        if (!currentUser.email) return;
                        fetch(`${BACKEND}/forgot-password`, {
                          method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ email: currentUser.email }),
                        }).catch(() => {});
                        setNotification(`If ${currentUser.email} is registered, a password reset link is on its way.`);
                      }}
                      title="Email yourself a link to set a new password"
                      style={{
                        display: "flex", alignItems: "center", gap: "6px", whiteSpace: "nowrap",
                        background: "#f8fafc", border: "1px solid #e5e7eb", borderRadius: "8px",
                        padding: "7px 12px", cursor: "pointer", fontSize: "12px", fontWeight: 500,
                        color: "#4b5563", fontFamily: UI_FONT,
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = "#f1f5f9"}
                      onMouseLeave={e => e.currentTarget.style.background = "#f8fafc"}
                    ><IconLock width={13} height={13} />Reset Password</button>
                  </div>
                </div>

                <div style={{ marginBottom: "20px" }}>
                  <div style={{ fontSize: "18px", fontWeight: 700, color: "#111827", marginBottom: "4px" }}>Welcome back, {currentUser.username}</div>
                  <div style={{ fontSize: "12px", color: "#6b7280" }}>Your personal activity across documents, chats, and task tools.</div>
                </div>

                {/* quick actions — this is what differentiates a personal dashboard
                    from the admin one: it's a launchpad, not a control panel */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", marginBottom: "20px" }}>
                  {[
                    { label: "Upload a Document", desc: "Add a new file to chat with.", icon: IconUpload, gradient: "linear-gradient(135deg,#2563eb,#1d4ed8)", onClick: () => fileInputRef.current?.click() },
                    { label: "Ask a Question", desc: "Jump back into Research & Q&A.", icon: IconFile, gradient: "linear-gradient(135deg,#059669,#047857)", onClick: () => { setActiveTask("research"); setMainView("chat"); } },
                    { label: "Browse Knowledge Base", desc: `${meStats?.kb_total ?? kbCount} verified answers firm-wide.`, icon: IconDatabase, gradient: "linear-gradient(135deg,#7c3aed,#6d28d9)", onClick: () => { setMainView("kb"); fetchKbEntries(); } },
                  ].map(a => (
                    <button key={a.label} onClick={a.onClick} style={{
                      textAlign: "left", border: "none", borderRadius: "16px", padding: "16px",
                      background: a.gradient, color: "#fff", cursor: "pointer",
                      display: "flex", flexDirection: "column", gap: "24px",
                      boxShadow: "0 3px 10px rgba(0,0,0,.08)", transition: "transform .15s", fontFamily: UI_FONT,
                    }}
                      onMouseEnter={e => e.currentTarget.style.transform = "translateY(-2px)"}
                      onMouseLeave={e => e.currentTarget.style.transform = "translateY(0)"}
                    >
                      <div style={{ width: "28px", height: "28px", borderRadius: "8px", background: "rgba(255,255,255,.22)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <a.icon width={15} height={15} />
                      </div>
                      <div>
                        <div style={{ fontSize: "13.5px", fontWeight: 700, marginBottom: "3px" }}>{a.label}</div>
                        <div style={{ fontSize: "11px", opacity: .88 }}>{a.desc}</div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* task-mode shortcuts — the other five tools, one tap away */}
                <div style={{ marginBottom: "20px" }}>
                  <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: "8px" }}>Jump Into a Tool</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                    {TASK_MODES.filter(t => t.id !== "research").map(t => (
                      <button key={t.id} onClick={() => { setActiveTask(t.id); setMainView("chat"); }} style={{
                        display: "flex", alignItems: "center", gap: "8px",
                        background: "#fff", border: "1px solid #eef0f3", borderRadius: "999px",
                        padding: "8px 14px", cursor: "pointer", fontFamily: UI_FONT, fontSize: "12.5px",
                        fontWeight: 600, color: "#374151", boxShadow: CARD_SHADOW, transition: "border-color .12s",
                      }}
                        onMouseEnter={e => e.currentTarget.style.borderColor = t.color}
                        onMouseLeave={e => e.currentTarget.style.borderColor = "#eef0f3"}
                      >
                        <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: t.color, flexShrink: 0 }} />
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>

                {meStatsLoading && !meStats && <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "10px" }}>Loading…</div>}

                {meStats && (() => {
                  const ratedPct = meStats.total_chats > 0 ? Math.round((meStats.rated_count / meStats.total_chats) * 100) : 0;
                  const unratedCount = Math.max(0, meStats.total_chats - meStats.rated_count);
                  const unratedRecent = (meStats.recent_chats || []).filter(c => !c.rating);
                  return (
                    <>
                      {/* unrated-answers nudge — makes the "% rated" stat actionable */}
                      {unratedCount > 0 && unratedRecent.length > 0 && (
                        <div style={{
                          display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px",
                          background: "#fffbeb", border: "1px solid #fde68a", borderRadius: "12px", padding: "12px 16px",
                        }}>
                          <IconThumbUp width={15} height={15} color="#d97706" style={{ flexShrink: 0 }} />
                          <div style={{ fontSize: "12.5px", color: "#92400e" }}>
                            <strong>{unratedCount}</strong> of your {meStats.total_chats} answer{meStats.total_chats === 1 ? "" : "s"} {unratedCount === 1 ? "isn't" : "aren't"} rated yet — rate the recent ones below to help improve future answers.
                          </div>
                        </div>
                      )}

                      {/* personal stat row */}
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", marginBottom: "16px" }}>
                        {[
                          { label: "My Documents", value: meStats.total_documents, icon: IconFile, tint: "#0891b2" },
                          { label: "My Chats", value: meStats.total_chats, icon: IconLayers, tint: "#2563eb", sub: `${meStats.chats_last_7d} in last 7 days` },
                          { label: "Avg Rating I've Given", value: meStats.average_rating ? `${meStats.average_rating}/5` : "—", icon: IconThumbUp, tint: "#d97706", sub: `${ratedPct}% of chats rated` },
                        ].map(card => (
                          <div key={card.label} style={{
                            background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "16px",
                            boxShadow: CARD_SHADOW,
                          }}>
                            <div style={{
                              width: "34px", height: "34px", borderRadius: "10px", background: card.tint, color: "#fff",
                              display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "12px",
                              boxShadow: `0 3px 10px ${card.tint}40`,
                            }}><card.icon width={16} height={16} /></div>
                            <div style={{ fontSize: "10.5px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", fontWeight: 700, marginBottom: "4px" }}>{card.label}</div>
                            <div style={{ fontSize: "22px", fontWeight: 700, color: "#111827" }}>{card.value ?? "—"}</div>
                            {card.sub && <div style={{ fontSize: "10.5px", color: "#9ca3af", marginTop: "2px" }}>{card.sub}</div>}
                          </div>
                        ))}
                      </div>

                      {/* activity + task-mode breakdown */}
                      <div style={{ display: "grid", gridTemplateColumns: "1.7fr 1fr", gap: "12px", marginBottom: "16px" }}>
                        <ChartCard title="My Activity" subtitle="Your chat volume, last 7 days">
                          {meStats.chats_per_day?.some(d => d.count > 0)
                            ? <ComboBarLine data={meStats.chats_per_day} height={170} barColor="#2563eb" />
                            : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>Ask a question to get started.</div>}
                        </ChartCard>
                        <ChartCard title="My Task Mode Usage" subtitle="Which tools you reach for">
                          {meStats.task_mode_usage?.length
                            ? <CategoricalBarList data={meStats.task_mode_usage.map(t => ({ label: t.label, value: t.count }))} />
                            : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No task-mode activity yet.</div>}
                        </ChartCard>
                      </div>

                      {/* recent chats — quick jump back in, rate inline */}
                      <ChartCard title="Recent Chats" subtitle="Your last few questions">
                        {(!meStats.recent_chats || meStats.recent_chats.length === 0) ? (
                          <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No chats yet — ask your first question to get started.</div>
                        ) : (
                          <div>
                            {meStats.recent_chats.map((chat, i) => (
                              <div key={chat.id} style={{
                                display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "12px",
                                padding: "10px 0", borderTop: i === 0 ? "none" : "1px solid #f4f5f7",
                              }}>
                                <div style={{ fontSize: "12.5px", color: "#374151", lineHeight: "1.4", flex: 1, minWidth: 0, ...clampStyle(1) }}>{chat.question}</div>
                                <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
                                  {chat.rating ? (
                                    <span style={{ fontSize: "11px", color: "#d97706" }}>{"★".repeat(chat.rating)}{"☆".repeat(5 - chat.rating)}</span>
                                  ) : (
                                    <div style={{ display: "flex", alignItems: "center", gap: "3px" }}>
                                      <button
                                        onClick={() => submitDashboardRating(chat.id, 5)}
                                        title="Good answer"
                                        style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", padding: "3px", display: "flex", borderRadius: "5px" }}
                                        onMouseEnter={e => { e.currentTarget.style.background = "#f3f4f6"; e.currentTarget.style.color = "#16a34a"; }}
                                        onMouseLeave={e => { e.currentTarget.style.background = "none"; e.currentTarget.style.color = "#9ca3af"; }}
                                      ><IconThumbUp width={12} height={12} /></button>
                                      <button
                                        onClick={() => submitDashboardRating(chat.id, 1)}
                                        title="Poor answer"
                                        style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", padding: "3px", display: "flex", borderRadius: "5px" }}
                                        onMouseEnter={e => { e.currentTarget.style.background = "#f3f4f6"; e.currentTarget.style.color = "#dc2626"; }}
                                        onMouseLeave={e => { e.currentTarget.style.background = "none"; e.currentTarget.style.color = "#9ca3af"; }}
                                      ><IconThumbDown width={12} height={12} /></button>
                                    </div>
                                  )}
                                  <span style={{ fontSize: "10.5px", color: "#c3c2b7", whiteSpace: "nowrap" }}>{timeAgo(chat.ts)}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </ChartCard>

                      {/* my documents — surfaces the Document Insights extraction that
                          already runs on every upload, previously admin-only to view */}
                      <div style={{ marginTop: "16px" }}>
                        <ChartCard title="My Documents" subtitle="Files you've uploaded, and what the AI found in them">
                          {(!meStats.my_documents || meStats.my_documents.length === 0) ? (
                            <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No documents yet — upload one to get started.</div>
                          ) : (
                            <div>
                              {meStats.my_documents.map((doc, i) => (
                                <div key={doc.id} style={{ padding: "10px 0", borderTop: i === 0 ? "none" : "1px solid #f4f5f7" }}>
                                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: "8px", minWidth: 0 }}>
                                      <div style={{
                                        width: "24px", height: "24px", borderRadius: "7px", background: "#eef2ff", color: "#4f46e5",
                                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                      }}><IconFile width={12} height={12} /></div>
                                      <div style={{ fontSize: "12.5px", fontWeight: 600, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{doc.filename}</div>
                                    </div>
                                    {doc.doc_type
                                      ? <Pill tone="corrected">{doc.doc_type}</Pill>
                                      : <Pill tone="neutral">Not analyzed yet</Pill>}
                                  </div>
                                  <div style={{ fontSize: "10.5px", color: "#9ca3af", marginLeft: "32px" }}>{timeAgo(doc.upload_timestamp)} · {doc.chunks} chunks</div>
                                  {doc.keywords?.length > 0 && (
                                    <div style={{ display: "flex", flexWrap: "wrap", gap: "5px", marginTop: "6px", marginLeft: "32px" }}>
                                      {doc.keywords.map((k, ki) => (
                                        <span key={ki} style={{ background: "#f0fdf4", color: "#166534", borderRadius: "999px", padding: "2px 8px", fontSize: "10px", fontWeight: 600 }}>{k}</span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </ChartCard>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#9ca3af", fontSize: "13px" }}>
              Sign in to view your dashboard.
            </div>
          )
        )}

        {/* ── ADMIN DASHBOARD ── */}
        {mainView === "admin" && (
          currentUser?.is_admin ? (
            <div style={{ flex: 1, overflow: "auto", padding: "26px 32px" }}>
              <div style={{ maxWidth: "1360px", margin: "0 auto" }}>

                {/* compact identity chip — a small right-aligned element like a real
                    product top nav (CRMi's "name + role beside a small avatar"),
                    not a full-width banner with mostly empty space */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
                  <div style={{ fontSize: "12px", color: "#9ca3af" }}>
                    {new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" })}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: "13px", fontWeight: 700, color: "#111827", lineHeight: "1.3" }}>{currentUser.username}</div>
                      <div style={{ fontSize: "10px", fontWeight: 700, color: "#7c3aed", letterSpacing: ".05em", textTransform: "uppercase" }}>Administrator</div>
                    </div>
                    <div style={{
                      width: "38px", height: "38px", borderRadius: "50%", flexShrink: 0,
                      background: "linear-gradient(135deg,#a78bfa,#7c3aed)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: "#fff", fontSize: "13px", fontWeight: 700,
                      boxShadow: "0 2px 8px rgba(124,58,237,.3)",
                    }}>{(currentUser.username || "?").slice(0, 2).toUpperCase()}</div>
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "12px", marginBottom: "18px" }}>
                  <div>
                    <div style={{ fontSize: "18px", fontWeight: 700, color: "#111827", marginBottom: "4px" }}>Admin Dashboard</div>
                    <div style={{ fontSize: "12px", color: "#6b7280" }}>Usage overview and full chat history across every user.</div>
                  </div>
                  <div style={{ display: "flex", gap: "3px", background: "#f1f5f9", padding: "3px", borderRadius: "10px" }}>
                    {[{ id: "overview", label: "Overview" }, { id: "users", label: "Users" }, { id: "history", label: "Chat History" }, { id: "insights", label: "Insights" }, { id: "documents", label: "Documents" }, { id: "audit", label: "Audit Log" }].map(t => (
                      <button key={t.id} onClick={() => setAdminTab(t.id)} style={{
                        padding: "6px 14px", borderRadius: "8px", border: "none", cursor: "pointer",
                        fontSize: "12.5px", fontFamily: UI_FONT, fontWeight: 600,
                        background: adminTab === t.id ? "#fff" : "transparent",
                        color: adminTab === t.id ? "#111827" : "#64748b",
                        boxShadow: adminTab === t.id ? "0 1px 3px rgba(0,0,0,.1)" : "none",
                        transition: "background .12s, color .12s",
                      }}>{t.label}</button>
                    ))}
                  </div>
                </div>

                {adminLoading && !adminOverview && <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "10px" }}>Loading…</div>}

                {/* ── OVERVIEW TAB — page composition modeled on a real admin dashboard:
                     ring-gauge stat cards + activity timeline, a hero chart with mini
                     stat-boxes beneath it, a donut with its legend below, then breakdowns ── */}
                {adminTab === "overview" && (() => {
                  const ratedCount = adminOverview?.rated_count || 0;
                  const totalChats = adminOverview?.total_chats || 0;
                  const dist = adminAnalytics?.rating_distribution || {};
                  const sentimentData = [
                    { label: "Positive (4-5★)", value: (dist["5"] || 0) + (dist["4"] || 0) },
                    { label: "Neutral (3★)", value: dist["3"] || 0 },
                    { label: "Negative (1-2★)", value: (dist["2"] || 0) + (dist["1"] || 0) },
                    { label: "Unrated", value: Math.max(0, totalChats - ratedCount) },
                  ].filter(d => d.value > 0);

                  const sumLast7 = series => (series || []).slice(-7).reduce((s, d) => s + d.count, 0);
                  const activeUsers = adminUsers.filter(u => u.is_active).length;
                  const usersTotal = adminUsers.length || adminOverview?.total_users || 0;
                  const chatsLast7 = sumLast7(adminAnalytics?.chats_per_day);
                  const docsLast7 = sumLast7(adminAnalytics?.documents_per_day);

                  const ACTIVITY_META = {
                    grant_admin:     { text: "granted admin access to", color: "#7c3aed" },
                    revoke_admin:    { text: "revoked admin access from", color: "#9ca3af" },
                    suspend_user:    { text: "suspended", color: "#d97706" },
                    reactivate_user: { text: "reactivated", color: "#16a34a" },
                    delete_user:     { text: "deleted", color: "#dc2626" },
                  };

                  return (
                    <>
                      {/* headline row — 3 ring-gauge stat cards + an activity timeline */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1.3fr", gap: "12px", marginBottom: "16px" }}>
                        <RingStatCard icon={IconUser} tint="#4f46e5" label="Users" sublabel={`${activeUsers} active now`}
                          total={usersTotal} ringValue={activeUsers} ringMax={usersTotal || 1} />
                        <RingStatCard icon={IconLayers} tint="#2563eb" label="Chats" sublabel={`${chatsLast7} in last 7 days`}
                          total={totalChats} ringValue={chatsLast7} ringMax={totalChats || 1} />
                        <RingStatCard icon={IconFile} tint="#0891b2" label="Documents" sublabel={`${docsLast7} in last 7 days`}
                          total={adminOverview?.total_documents ?? 0} ringValue={docsLast7} ringMax={(adminOverview?.total_documents || 1)} />

                        <ChartCard title="Recent Activity">
                          {adminAuditLog.length === 0 ? (
                            <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No admin actions recorded yet.</div>
                          ) : (
                            <div>
                              {adminAuditLog.slice(0, 5).map((entry, i) => {
                                const meta = ACTIVITY_META[entry.action] || { text: entry.action, color: "#6b7280" };
                                const isLast = i === Math.min(adminAuditLog.length, 5) - 1;
                                const time = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
                                return (
                                  <div key={i} style={{ display: "flex", gap: "10px" }}>
                                    <div style={{ fontSize: "10.5px", fontWeight: 700, color: "#111827", width: "40px", flexShrink: 0, paddingTop: "1px" }}>{time}</div>
                                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                                      <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: meta.color, marginTop: "3px", flexShrink: 0 }} />
                                      {!isLast && <span style={{ flex: 1, width: "1px", background: "#eef0f3", marginTop: "2px" }} />}
                                    </div>
                                    <div style={{ fontSize: "11px", color: "#374151", lineHeight: "1.5", paddingBottom: isLast ? 0 : "13px" }}>
                                      <strong>{entry.actor}</strong> {meta.text} <strong>{entry.target}</strong>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </ChartCard>
                      </div>

                      {/* hero: chat activity chart with mini stat-boxes beneath it, paired with a donut */}
                      <div style={{ display: "grid", gridTemplateColumns: "1.7fr 1fr", gap: "12px", marginBottom: "16px" }}>
                        <ChartCard title="Chat Activity" subtitle="Daily volume with a 3-day trend line, last 14 days">
                          {adminAnalytics ? (
                            <>
                              <ComboBarLine data={adminAnalytics.chats_per_day} height={170} barColor="#2563eb" />
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px", marginTop: "16px" }}>
                                {[
                                  { label: "Knowledge Base", value: adminOverview?.total_kb_entries ?? "—", tint: "#059669" },
                                  { label: "Avg Rating", value: adminOverview?.average_rating ? `${adminOverview.average_rating}/5` : "—", tint: "#d97706" },
                                  { label: "Rated", value: `${ratedCount}/${totalChats}`, tint: "#7c3aed" },
                                ].map(s => (
                                  <div key={s.label} style={{ background: "#f8fafc", borderRadius: "10px", padding: "11px 14px" }}>
                                    <div style={{ fontSize: "17px", fontWeight: 700, color: s.tint }}>{s.value}</div>
                                    <div style={{ fontSize: "10.5px", color: "#6b7280", marginTop: "2px" }}>{s.label}</div>
                                  </div>
                                ))}
                              </div>
                            </>
                          ) : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No data yet.</div>}
                        </ChartCard>

                        <ChartCard title="Chats by Sentiment" subtitle="Rated 4-5★ vs. 1-2★ vs. never rated">
                          {sentimentData.length > 0
                            ? <div style={{ display: "flex", justifyContent: "center", paddingTop: "10px" }}><DonutChart data={sentimentData} layout="column" /></div>
                            : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No chats yet.</div>}
                        </ChartCard>
                      </div>

                      {/* when it happens + is everything healthy */}
                      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: "12px", marginBottom: "16px" }}>
                        <ChartCard
                          title="Activity heatmap"
                          subtitle={adminAnalytics?.peak_insight ? `Busiest: ${adminAnalytics.peak_insight}` : "When chats happen, by day and hour"}
                        >
                          {adminAnalytics && adminAnalytics.activity_heatmap?.some(c => c.count > 0)
                            ? <div style={{ maxWidth: "480px", overflowX: "auto" }}><ActivityHeatmap cells={adminAnalytics.activity_heatmap} /></div>
                            : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No activity recorded yet.</div>}
                        </ChartCard>
                        <ChartCard title="System status" action={
                          <button onClick={() => fetchAdminData(adminChatPage, { search: adminSearch, rating: historyRating, days: historyDays, userId: historyUserId })} style={{
                            background: "none", border: "none", cursor: "pointer", color: "#9ca3af", display: "flex",
                          }}><IconRefresh width={13} height={13} /></button>
                        }>
                          {adminSystemStatus ? (
                            <div>
                              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: adminSystemStatus.database.ok ? "#0ca30c" : "#d03b3b", flexShrink: 0 }} />
                                <span style={{ fontSize: "12px", color: "#374151", fontWeight: 600 }}>Database</span>
                                <span style={{ fontSize: "11px", color: "#9ca3af", marginLeft: "auto" }}>{adminSystemStatus.database.ok ? "Connected" : "Unreachable"}</span>
                              </div>
                              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: adminSystemStatus.llm_backend.ok ? "#0ca30c" : "#d03b3b", flexShrink: 0 }} />
                                <span style={{ fontSize: "12px", color: "#374151", fontWeight: 600 }}>LLM backend (Ollama)</span>
                                <span style={{ fontSize: "11px", color: "#9ca3af", marginLeft: "auto" }}>{adminSystemStatus.llm_backend.ok ? "Reachable" : "Unreachable"}</span>
                              </div>
                              {adminSystemStatus.llm_backend.models?.length > 0 && (
                                <div style={{ paddingLeft: "16px", marginBottom: "8px" }}>
                                  {adminSystemStatus.llm_backend.models.map(m => (
                                    <div key={m.name} style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#6b7280", padding: "2px 0" }}>
                                      <span style={{ fontFamily: "'Courier New',monospace" }}>{m.name}</span>
                                      <span>{m.size_gb} GB</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: adminSystemStatus.openrouter_fallback_configured ? "#0ca30c" : "#c3c2b7", flexShrink: 0 }} />
                                <span style={{ fontSize: "12px", color: "#374151", fontWeight: 600 }}>OpenRouter fallback</span>
                                <span style={{ fontSize: "11px", color: "#9ca3af", marginLeft: "auto" }}>{adminSystemStatus.openrouter_fallback_configured ? "Configured" : "Not configured"}</span>
                              </div>
                            </div>
                          ) : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>Checking…</div>}
                        </ChartCard>
                      </div>

                      {/* breakdowns — by task mode, by user */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                        <ChartCard title="Usage by Task Mode">
                          {adminAnalytics?.task_mode_usage?.length
                            ? <CategoricalBarList data={adminAnalytics.task_mode_usage.map(t => ({ label: t.label, value: t.count }))} />
                            : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No data yet.</div>}
                        </ChartCard>
                        <ChartCard title="Most Active Users">
                          {(adminAnalytics?.top_users || []).length === 0 ? (
                            <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No signed-in activity yet.</div>
                          ) : (
                            <div>
                              {adminAnalytics.top_users.map((u, i) => (
                                <div key={u.username} style={{
                                  display: "flex", alignItems: "center", gap: "10px", padding: "8px 0",
                                  borderTop: i === 0 ? "none" : "1px solid #f4f5f7",
                                }}>
                                  <div style={{
                                    width: "30px", height: "30px", borderRadius: "50%", flexShrink: 0,
                                    background: CATEGORICAL[i % CATEGORICAL.length],
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    color: "#fff", fontSize: "11px", fontWeight: 700,
                                  }}>{u.username.slice(0, 2).toUpperCase()}</div>
                                  <div style={{ flex: 1, minWidth: 0, fontSize: "12.5px", fontWeight: 600, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.username}</div>
                                  <div style={{ textAlign: "right", flexShrink: 0 }}>
                                    <div style={{ fontSize: "12.5px", fontWeight: 700, color: "#111827" }}>{u.chat_count}</div>
                                    <div style={{ fontSize: "9.5px", color: "#9ca3af" }}>chats</div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </ChartCard>
                      </div>
                    </>
                  );
                })()}

                {/* ── USERS TAB ── */}
                {adminTab === "users" && (() => {
                  const filtered = adminUsers.filter(u =>
                    !userSearch ||
                    u.username?.toLowerCase().includes(userSearch.toLowerCase()) ||
                    u.email?.toLowerCase().includes(userSearch.toLowerCase())
                  );
                  return (
                    <>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px", gap: "10px" }}>
                        <div>
                          <div style={{ fontSize: "14px", fontWeight: 700, color: "#111827" }}>Users</div>
                          <div style={{ fontSize: "11px", color: "#9ca3af" }}>{adminUsers.length} accounts · view activity, roles, and access</div>
                        </div>
                        <div style={{ display: "flex", gap: "8px" }}>
                          <input
                            value={userSearch}
                            onChange={e => setUserSearch(e.target.value)}
                            placeholder="Search name or email…"
                            style={{
                              border: "1px solid #d1d5db", borderRadius: "8px", padding: "6px 10px",
                              fontSize: "12.5px", fontFamily: UI_FONT, outline: "none", width: "220px",
                            }}
                          />
                          <button onClick={fetchAdminUsers} title="Refresh" style={{
                            background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px",
                            padding: "6px 10px", cursor: "pointer", display: "flex", color: "#6b7280",
                          }}><IconRefresh width={14} height={14} /></button>
                        </div>
                      </div>

                      <div style={{
                        background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", overflow: "hidden",
                        boxShadow: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)",
                      }}>
                        <div style={{
                          display: "grid", gridTemplateColumns: "2fr 0.8fr 0.8fr 0.6fr 0.6fr 1fr 1fr",
                          padding: "10px 16px", background: "#f8fafc", borderBottom: "1px solid #eef0f3",
                          fontSize: "10.5px", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".04em",
                        }}>
                          <span>User</span><span>Role</span><span>Status</span><span>Chats</span><span>Docs</span><span>Last Active</span><span>Actions</span>
                        </div>

                        {adminUsersLoading && <div style={{ padding: "18px", fontSize: "12px", color: "#94a3b8" }}>Loading…</div>}
                        {!adminUsersLoading && filtered.length === 0 && (
                          <div style={{ padding: "18px", fontSize: "12.5px", color: "#64748b", textAlign: "center" }}>No matching users.</div>
                        )}
                        {filtered.map(u => (
                          <div key={u.id} style={{
                            display: "grid", gridTemplateColumns: "2fr 0.8fr 0.8fr 0.6fr 0.6fr 1fr 1fr",
                            padding: "11px 16px", borderTop: "1px solid #f4f5f7", alignItems: "center",
                            fontSize: "12.5px", cursor: "pointer",
                          }}
                            onClick={() => openUserDetail(u)}
                            onMouseEnter={e => e.currentTarget.style.background = "#fafbfc"}
                            onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                          >
                            <div style={{ display: "flex", alignItems: "center", gap: "9px", minWidth: 0 }}>
                              <div style={{
                                width: "26px", height: "26px", borderRadius: "50%", flexShrink: 0,
                                background: u.is_admin ? "linear-gradient(135deg,#a78bfa,#7c3aed)" : "linear-gradient(135deg,#60a5fa,#1d4ed8)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                                color: "#fff", fontSize: "10px", fontWeight: 700,
                              }}>{(u.username || "?").slice(0, 2).toUpperCase()}</div>
                              <div style={{ minWidth: 0 }}>
                                <div style={{ fontWeight: 600, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.username}</div>
                                <div style={{ fontSize: "10.5px", color: "#9ca3af", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.email}</div>
                              </div>
                            </div>
                            <span><Pill tone={u.is_admin ? "admin" : "user"}>{u.is_admin ? "Admin" : "User"}</Pill></span>
                            <span><Pill tone={u.is_active ? "active" : "suspended"}>{u.is_active ? "Active" : "Suspended"}</Pill></span>
                            <span style={{ color: "#374151" }}>{u.chat_count}</span>
                            <span style={{ color: "#374151" }}>{u.document_count}</span>
                            <span style={{ color: "#9ca3af", fontSize: "11px" }}>
                              {u.last_active ? new Date(u.last_active).toLocaleDateString() : "Never"}
                            </span>
                            <span style={{ display: "flex", gap: "4px" }} onClick={e => e.stopPropagation()}>
                              <button title={u.is_active ? "Suspend" : "Reactivate"}
                                onClick={() => u.is_active
                                  ? setConfirmAction({ type: "suspend", user: u })
                                  : updateUser(u.id, { is_active: true })}
                                style={{ background: "none", border: "none", cursor: "pointer", color: u.is_active ? "#d97706" : "#16a34a", padding: "5px", borderRadius: "6px", display: "flex" }}
                              >{u.is_active ? <IconBan width={14} height={14} /> : <IconCheckCircle width={14} height={14} />}</button>
                              <button title="Delete user"
                                onClick={() => setConfirmAction({ type: "delete", user: u })}
                                style={{ background: "none", border: "none", cursor: "pointer", color: "#dc2626", padding: "5px", borderRadius: "6px", display: "flex" }}
                              ><IconTrash width={14} height={14} /></button>
                              <button title="View details" onClick={() => openUserDetail(u)}
                                style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", padding: "5px", borderRadius: "6px", display: "flex" }}
                              ><IconChevronRight width={14} height={14} /></button>
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  );
                })()}

                {/* ── CHAT HISTORY TAB ── */}
                {adminTab === "history" && (() => {
                  const toggleChatExpand = id => setExpandedChats(prev => {
                    const next = new Set(prev);
                    next.has(id) ? next.delete(id) : next.add(id);
                    return next;
                  });
                  const dayGroups = groupByDay(adminChats);
                  const filtersActive = adminSearch || historyRating || historyDays || historyUserId;
                  const exportHistory = async () => {
                    const qs = new URLSearchParams({ token: authToken, limit: "1000", offset: "0" });
                    if (adminSearch) qs.set("search", adminSearch);
                    if (historyRating) qs.set("rating", historyRating);
                    if (historyDays) qs.set("since_days", historyDays);
                    if (historyUserId) qs.set("user_id", historyUserId);
                    try {
                      const res = await fetch(`${BACKEND}/admin/chat_history?${qs.toString()}`);
                      const data = await res.json();
                      downloadCsv(
                        `chat_history_${new Date().toISOString().slice(0, 10)}.csv`,
                        ["User", "Timestamp", "Question", "Answer", "Rating"],
                        (data.items || []).map(c => [c.username, c.timestamp, c.question, c.answer, c.rating ?? ""])
                      );
                    } catch { setNotification("Could not export chat history"); }
                  };

                  return (
                    <>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px", gap: "10px", flexWrap: "wrap" }}>
                        <div>
                          <div style={{ fontSize: "14px", fontWeight: 700, color: "#111827" }}>Chat History</div>
                          <div style={{ fontSize: "11px", color: "#9ca3af" }}>{adminChatTotal} conversations across every user</div>
                        </div>
                        <button onClick={exportHistory} style={{
                          display: "flex", alignItems: "center", gap: "6px",
                          background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px",
                          padding: "7px 12px", cursor: "pointer", color: "#374151",
                          fontSize: "12px", fontFamily: UI_FONT, fontWeight: 600,
                        }}><IconDownload width={13} height={13} />Export CSV</button>
                      </div>

                      {/* filter bar */}
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "14px" }}>
                        <div style={{ position: "relative", flex: "1 1 200px" }}>
                          <div style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "#9ca3af", display: "flex" }}>
                            <IconSearch width={13} height={13} />
                          </div>
                          <input
                            value={adminSearch}
                            onChange={e => { setAdminSearch(e.target.value); setAdminChatPage(0); }}
                            placeholder="Search questions & answers…"
                            style={{
                              width: "100%", border: "1px solid #d1d5db", borderRadius: "8px", padding: "7px 10px 7px 30px",
                              fontSize: "12.5px", fontFamily: UI_FONT, outline: "none", boxSizing: "border-box",
                            }}
                          />
                        </div>
                        <select value={historyUserId} onChange={e => { setHistoryUserId(e.target.value); setAdminChatPage(0); }} style={{
                          border: "1px solid #d1d5db", borderRadius: "8px", padding: "0 10px",
                          fontSize: "12.5px", fontFamily: UI_FONT, color: "#374151", background: "#fff", cursor: "pointer",
                        }}>
                          <option value="">All users</option>
                          {adminUsers.map(u => <option key={u.id} value={u.id}>{u.username}</option>)}
                        </select>
                        <select value={historyRating} onChange={e => { setHistoryRating(e.target.value); setAdminChatPage(0); }} style={{
                          border: "1px solid #d1d5db", borderRadius: "8px", padding: "0 10px",
                          fontSize: "12.5px", fontFamily: UI_FONT, color: "#374151", background: "#fff", cursor: "pointer",
                        }}>
                          <option value="">All ratings</option>
                          <option value="5">★★★★★</option>
                          <option value="4">★★★★</option>
                          <option value="3">★★★</option>
                          <option value="2">★★</option>
                          <option value="1">★</option>
                          <option value="unrated">Unrated</option>
                        </select>
                        <select value={historyDays} onChange={e => { setHistoryDays(e.target.value); setAdminChatPage(0); }} style={{
                          border: "1px solid #d1d5db", borderRadius: "8px", padding: "0 10px",
                          fontSize: "12.5px", fontFamily: UI_FONT, color: "#374151", background: "#fff", cursor: "pointer",
                        }}>
                          <option value="">All time</option>
                          <option value="1">Today</option>
                          <option value="7">Last 7 days</option>
                          <option value="30">Last 30 days</option>
                        </select>
                        {filtersActive && (
                          <button onClick={() => { setAdminSearch(""); setHistoryRating(""); setHistoryDays(""); setHistoryUserId(""); setAdminChatPage(0); }} style={{
                            background: "none", border: "none", cursor: "pointer", color: "#2563eb",
                            fontSize: "12px", fontFamily: UI_FONT, fontWeight: 600,
                          }}>Clear filters</button>
                        )}
                      </div>

                      <div style={{
                        background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", overflow: "hidden",
                        boxShadow: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05)",
                      }}>
                        {adminChats.length === 0 && !adminLoading && (
                          <div style={{ padding: "30px", fontSize: "12.5px", color: "#64748b", textAlign: "center" }}>No chat history matches these filters.</div>
                        )}
                        {dayGroups.map((group, gi) => (
                          <div key={group.date}>
                            <div style={{
                              padding: "8px 16px", background: "#f8fafc",
                              borderTop: gi === 0 ? "none" : "1px solid #eef0f3", borderBottom: "1px solid #eef0f3",
                              fontSize: "10.5px", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".04em",
                            }}>{formatDayHeader(group.date)}</div>
                            {group.items.map((chat, i) => {
                              const isOpen = expandedChats.has(chat.id);
                              return (
                                <div key={chat.id}
                                  onClick={() => toggleChatExpand(chat.id)}
                                  style={{
                                    padding: "12px 16px", background: "#fff", cursor: "pointer",
                                    borderTop: i === 0 ? "none" : "1px solid #f4f5f7",
                                  }}
                                  onMouseEnter={e => e.currentTarget.style.background = "#fafbfc"}
                                  onMouseLeave={e => e.currentTarget.style.background = "#fff"}
                                >
                                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px", gap: "8px" }}>
                                    <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontWeight: 600, color: "#374151", minWidth: 0 }}>
                                      <div style={{
                                        width: "18px", height: "18px", borderRadius: "50%", flexShrink: 0,
                                        background: "linear-gradient(135deg,#60a5fa,#1d4ed8)", color: "#fff",
                                        display: "flex", alignItems: "center", justifyContent: "center", fontSize: "8px", fontWeight: 700,
                                      }}>{(chat.username || "?").slice(0, 2).toUpperCase()}</div>
                                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{chat.username}</span>
                                    </span>
                                    <span style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
                                      {chat.rating != null && <Pill tone="rated">{"★".repeat(chat.rating)}</Pill>}
                                      <span style={{ fontSize: "10.5px", color: "#c3c2b7" }}>
                                        {chat.timestamp ? new Date(chat.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""}
                                      </span>
                                    </span>
                                  </div>
                                  <div style={{ fontSize: "12.5px", fontWeight: 600, color: "#1e293b", marginBottom: "4px", lineHeight: "1.4", ...(isOpen ? {} : clampStyle(2)) }}>
                                    {chat.question}
                                  </div>
                                  <div style={{ fontSize: "12px", color: "#475569", lineHeight: "1.5", ...(isOpen ? {} : clampStyle(2)) }}>
                                    {chat.answer}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>

                      {adminChatTotal > ADMIN_PAGE_SIZE && (
                        <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "12px", marginTop: "14px" }}>
                          <button
                            onClick={() => setAdminChatPage(p => Math.max(0, p - 1))}
                            disabled={adminChatPage === 0}
                            style={{
                              background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px",
                              padding: "6px 14px", fontSize: "12px", cursor: adminChatPage === 0 ? "default" : "pointer",
                              color: adminChatPage === 0 ? "#cbd5e1" : "#374151", fontFamily: UI_FONT, fontWeight: 600,
                            }}
                          >Previous</button>
                          <span style={{ fontSize: "11.5px", color: "#6b7280" }}>
                            Page {adminChatPage + 1} of {Math.max(1, Math.ceil(adminChatTotal / ADMIN_PAGE_SIZE))}
                          </span>
                          <button
                            onClick={() => setAdminChatPage(p => (p + 1) * ADMIN_PAGE_SIZE < adminChatTotal ? p + 1 : p)}
                            disabled={(adminChatPage + 1) * ADMIN_PAGE_SIZE >= adminChatTotal}
                            style={{
                              background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px",
                              padding: "6px 14px", fontSize: "12px",
                              cursor: (adminChatPage + 1) * ADMIN_PAGE_SIZE >= adminChatTotal ? "default" : "pointer",
                              color: (adminChatPage + 1) * ADMIN_PAGE_SIZE >= adminChatTotal ? "#cbd5e1" : "#374151", fontFamily: UI_FONT, fontWeight: 600,
                            }}
                          >Next</button>
                        </div>
                      )}
                    </>
                  );
                })()}

                {/* ── INSIGHTS TAB — facts, keywords, entities extracted from every
                     document by the LLM, aggregated for the admin dashboard. ── */}
                {adminTab === "insights" && (() => {
                  const toggleInsightDoc = id => setExpandedInsightDocs(prev => {
                    const next = new Set(prev);
                    next.has(id) ? next.delete(id) : next.add(id);
                    return next;
                  });
                  const docTypeData = (adminInsights?.doc_types || []).map(d => ({ label: d.label, value: d.count }));
                  const keywordData = (adminInsights?.top_keywords || []).map(k => ({ label: k.label, value: k.count }));

                  return (
                    <>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", gap: "10px", flexWrap: "wrap" }}>
                        <div>
                          <div style={{ fontSize: "14px", fontWeight: 700, color: "#111827" }}>Document Insights</div>
                          <div style={{ fontSize: "11px", color: "#9ca3af" }}>
                            Facts, keywords, and entities the LLM extracted from every uploaded document.
                          </div>
                        </div>
                        <button onClick={analyzePendingDocuments} disabled={insightsAnalyzeBusy || !adminInsights?.pending_count} style={{
                          display: "flex", alignItems: "center", gap: "6px",
                          background: (!adminInsights?.pending_count) ? "#f1f5f9" : "#2563eb",
                          border: "none", borderRadius: "8px",
                          padding: "8px 14px", cursor: (insightsAnalyzeBusy || !adminInsights?.pending_count) ? "default" : "pointer",
                          color: (!adminInsights?.pending_count) ? "#94a3b8" : "#fff",
                          fontSize: "12px", fontFamily: UI_FONT, fontWeight: 700,
                        }}>
                          <IconWand width={13} height={13} />
                          {insightsAnalyzeBusy ? "Queuing…" : adminInsights?.pending_count ? `Analyze ${adminInsights.pending_count} Pending` : "All Documents Analyzed"}
                        </button>
                      </div>

                      {adminInsightsLoading && !adminInsights && <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "10px" }}>Loading…</div>}

                      {adminInsights && (
                        <>
                          {/* stat row */}
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px", marginBottom: "16px" }}>
                            {[
                              { label: "Documents Analyzed", value: `${adminInsights.analyzed_count}/${adminInsights.total_documents}`, icon: IconFile, tint: "#0891b2" },
                              { label: "Facts Extracted", value: adminInsights.total_facts, icon: IconListChecks, tint: "#2563eb" },
                              { label: "Unique Keywords", value: adminInsights.unique_keywords, icon: IconTag, tint: "#059669" },
                              { label: "Named Entities", value: (adminInsights.top_entities || []).length, icon: IconUser, tint: "#7c3aed" },
                            ].map(card => (
                              <div key={card.label} style={{
                                background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "14px 16px",
                                boxShadow: CARD_SHADOW, display: "flex", alignItems: "center", justifyContent: "space-between",
                              }}>
                                <div>
                                  <div style={{ fontSize: "10px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", fontWeight: 700, marginBottom: "4px" }}>{card.label}</div>
                                  <div style={{ fontSize: "20px", fontWeight: 700, color: "#111827" }}>{card.value}</div>
                                </div>
                                <div style={{
                                  width: "30px", height: "30px", borderRadius: "9px", background: card.tint + "16", color: card.tint,
                                  display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                }}><card.icon width={15} height={15} /></div>
                              </div>
                            ))}
                          </div>

                          {/* top keywords + document types */}
                          <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: "12px", marginBottom: "16px" }}>
                            <ChartCard title="Top Keywords" subtitle="Most frequent terms across all analyzed documents">
                              {keywordData.length > 0
                                ? <CategoricalBarList data={keywordData} maxRows={10} />
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No keywords extracted yet — analyze documents to populate this.</div>}
                            </ChartCard>
                            <ChartCard title="Document Types" subtitle="Auto-classified by the LLM">
                              {docTypeData.length > 0
                                ? <div style={{ display: "flex", justifyContent: "center", paddingTop: "6px" }}><DonutChart data={docTypeData} layout="column" /></div>
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No documents classified yet.</div>}
                            </ChartCard>
                          </div>

                          {/* top entities */}
                          {(adminInsights.top_entities || []).length > 0 && (
                            <ChartCard title="Most-Mentioned Entities" subtitle="Parties, organizations, and people named across documents" >
                              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                                {adminInsights.top_entities.map((e, i) => (
                                  // Index, not e.label — these are LLM-extracted entity
                                  // names across many documents, so an exact duplicate
                                  // string is plausible (see DonutChart's note above).
                                  <span key={i} style={{
                                    display: "flex", alignItems: "center", gap: "6px",
                                    background: "#f8fafc", border: "1px solid #eef0f3", borderRadius: "999px",
                                    padding: "6px 12px", fontSize: "12px", color: "#374151", fontWeight: 600,
                                  }}>{e.label}<span style={{ color: "#94a3b8", fontWeight: 700 }}>{e.count}</span></span>
                                ))}
                              </div>
                            </ChartCard>
                          )}

                          <div style={{ height: "16px" }} />

                          {/* per-document browsable list */}
                          <div style={{ fontSize: "13px", fontWeight: 700, color: "#111827", marginBottom: "10px" }}>
                            Analyzed Documents
                            <span style={{ fontWeight: 500, color: "#9ca3af", marginLeft: "6px" }}>({adminInsights.documents.length})</span>
                          </div>

                          {adminInsights.documents.length === 0 ? (
                            <div style={{
                              padding: "40px 20px", background: "#fff", border: "1px solid #eef0f3",
                              borderRadius: "14px", textAlign: "center",
                            }}>
                              <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151", marginBottom: "4px" }}>No documents analyzed yet</div>
                              <div style={{ fontSize: "12px", color: "#9ca3af" }}>Click "Analyze Pending" above, or upload a new document — analysis runs automatically on upload.</div>
                            </div>
                          ) : (
                            <div style={{ display: "grid", gap: "10px" }}>
                              {adminInsights.documents.map(doc => {
                                const isOpen = expandedInsightDocs.has(doc.document_id);
                                return (
                                  <div key={doc.document_id} onClick={() => toggleInsightDoc(doc.document_id)} style={{
                                    background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "14px 16px",
                                    boxShadow: CARD_SHADOW, cursor: "pointer",
                                  }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px" }}>
                                      <div style={{ display: "flex", alignItems: "center", gap: "10px", minWidth: 0 }}>
                                        <div style={{
                                          width: "26px", height: "26px", borderRadius: "8px", background: "#eef2ff", color: "#4f46e5",
                                          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                        }}><IconFile width={13} height={13} /></div>
                                        <div style={{ fontSize: "12.5px", fontWeight: 700, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{doc.filename}</div>
                                        <Pill tone="neutral">{doc.doc_type || "Other"}</Pill>
                                      </div>
                                      <IconChevronRight width={14} height={14} color="#c3c2b7" style={{ transform: isOpen ? "rotate(90deg)" : "none", transition: "transform .12s", flexShrink: 0 }} />
                                    </div>

                                    {isOpen && (
                                      <div style={{ marginTop: "12px", paddingTop: "12px", borderTop: "1px solid #f4f5f7", display: "grid", gap: "12px" }}>
                                        {doc.keywords?.length > 0 && (
                                          <div>
                                            <div style={{ fontSize: "9.5px", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: "6px" }}>Keywords</div>
                                            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                                              {doc.keywords.map((k, i) => (
                                                <span key={i} style={{ background: "#f0fdf4", color: "#166534", borderRadius: "999px", padding: "3px 10px", fontSize: "11px", fontWeight: 600 }}>{k}</span>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                        {doc.facts?.length > 0 && (
                                          <div>
                                            <div style={{ fontSize: "9.5px", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: "6px" }}>Key Facts</div>
                                            <ul style={{ margin: 0, paddingLeft: "16px" }}>
                                              {doc.facts.map((f, i) => (
                                                <li key={i} style={{ fontSize: "12px", color: "#374151", lineHeight: "1.6" }}>{f}</li>
                                              ))}
                                            </ul>
                                          </div>
                                        )}
                                        {doc.entities?.length > 0 && (
                                          <div>
                                            <div style={{ fontSize: "9.5px", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: "6px" }}>Entities</div>
                                            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                                              {doc.entities.map((e, i) => (
                                                <span key={i} style={{ background: "#eef2ff", color: "#4338ca", borderRadius: "999px", padding: "3px 10px", fontSize: "11px", fontWeight: 600 }}>{e}</span>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                        <div style={{ fontSize: "10px", color: "#c3c2b7" }}>Analyzed {timeAgo(doc.analyzed_at)}</div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </>
                      )}
                    </>
                  );
                })()}

                {/* ── DOCUMENTS TAB — every document across every user; the admin-wide
                     view that didn't exist before (each user could only see their own). ── */}
                {adminTab === "documents" && (() => {
                  const totalPages = adminDocs ? Math.max(1, Math.ceil(adminDocs.total_count / ADMIN_PAGE_SIZE)) : 1;
                  return (
                    <>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", gap: "10px", flexWrap: "wrap" }}>
                        <div>
                          <div style={{ fontSize: "14px", fontWeight: 700, color: "#111827" }}>Documents</div>
                          <div style={{ fontSize: "11px", color: "#9ca3af" }}>Every document uploaded, by every user.</div>
                        </div>
                      </div>

                      {adminDocsLoading && !adminDocs && <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "10px" }}>Loading…</div>}

                      {adminDocs && (
                        <>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px", marginBottom: "16px" }}>
                            {[
                              { label: "Total Documents", value: adminDocs.total_documents, icon: IconFile, tint: "#0891b2" },
                              { label: "Analyzed", value: adminDocs.analyzed_count, icon: IconWand, tint: "#7c3aed" },
                              { label: "Unique Uploaders", value: adminDocs.unique_uploaders, icon: IconUser, tint: "#2563eb" },
                              { label: "Total Chunks", value: adminDocs.total_chunks, icon: IconLayers, tint: "#059669" },
                            ].map(card => (
                              <div key={card.label} style={{
                                background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", padding: "14px 16px",
                                boxShadow: CARD_SHADOW, display: "flex", alignItems: "center", justifyContent: "space-between",
                              }}>
                                <div>
                                  <div style={{ fontSize: "10px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", fontWeight: 700, marginBottom: "4px" }}>{card.label}</div>
                                  <div style={{ fontSize: "20px", fontWeight: 700, color: "#111827" }}>{card.value}</div>
                                </div>
                                <div style={{
                                  width: "30px", height: "30px", borderRadius: "9px", background: card.tint + "16", color: card.tint,
                                  display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                }}><card.icon width={15} height={15} /></div>
                              </div>
                            ))}
                          </div>

                          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 1fr", gap: "12px", marginBottom: "12px" }}>
                            <ChartCard title="Uploads Over Time" subtitle="Daily volume, last 14 days">
                              {adminAnalytics?.documents_per_day?.some(d => d.count > 0)
                                ? <ComboBarLine data={adminAnalytics.documents_per_day} height={170} barColor="#0891b2" />
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No uploads in this window.</div>}
                            </ChartCard>
                            <ChartCard title="Documents by Type" subtitle="Auto-classified by Insights">
                              {adminDocs.doc_types?.length
                                ? <div style={{ display: "flex", justifyContent: "center", paddingTop: "6px" }}><DonutChart data={adminDocs.doc_types.map(t => ({ label: t.label, value: t.count }))} layout="column" /></div>
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No documents classified yet.</div>}
                            </ChartCard>
                            <ChartCard title="Analysis Coverage" subtitle="Share put through Document Insights">
                              <div style={{ display: "flex", justifyContent: "center", padding: "6px 0 2px" }}>
                                <RadialProgress value={adminDocs.analyzed_count} max={adminDocs.total_documents || 1} color="#7c3aed"
                                  label="Analyzed" sublabel={`${adminDocs.analyzed_count} of ${adminDocs.total_documents}`} />
                              </div>
                            </ChartCard>
                          </div>

                          <div style={{ marginBottom: "16px" }}>
                            <ChartCard title="Top Uploaders">
                              {adminDocs.top_uploaders?.length
                                ? <CategoricalBarList data={adminDocs.top_uploaders.map(u => ({ label: u.label, value: u.count }))} />
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No documents yet.</div>}
                            </ChartCard>
                          </div>

                          <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
                            <div style={{ position: "relative", flex: 1 }}>
                              <div style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "#9ca3af", display: "flex" }}>
                                <IconSearch width={13} height={13} />
                              </div>
                              <input
                                value={adminDocsSearch}
                                onChange={e => { setAdminDocsSearch(e.target.value); setAdminDocsPage(0); }}
                                placeholder="Search filename or uploader…"
                                style={{
                                  width: "100%", border: "1px solid #d1d5db", borderRadius: "8px", padding: "7px 10px 7px 30px",
                                  fontSize: "12.5px", fontFamily: UI_FONT, outline: "none", boxSizing: "border-box",
                                }}
                              />
                            </div>
                            <button onClick={() => fetchAdminDocuments(adminDocsPage, adminDocsSearch)} title="Refresh" style={{
                              background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px",
                              padding: "0 10px", cursor: "pointer", display: "flex", alignItems: "center", color: "#6b7280",
                            }}><IconRefresh width={13} height={13} /></button>
                          </div>

                          {adminDocs.documents.length === 0 ? (
                            <div style={{ padding: "40px 20px", background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", textAlign: "center" }}>
                              <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}>No documents match.</div>
                            </div>
                          ) : (
                            <div style={{ background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", overflow: "hidden" }}>
                              {adminDocs.documents.map((doc, i) => (
                                <div key={doc.id} style={{
                                  display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px",
                                  borderTop: i === 0 ? "none" : "1px solid #f4f5f7",
                                }}>
                                  <div style={{
                                    width: "28px", height: "28px", borderRadius: "8px", background: "#eef2ff", color: "#4f46e5",
                                    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                  }}><IconFile width={14} height={14} /></div>
                                  <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: "12.5px", fontWeight: 700, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{doc.filename}</div>
                                    <div style={{ fontSize: "11px", color: "#9ca3af" }}>Uploaded by <strong style={{ color: "#6b7280" }}>{doc.uploaded_by}</strong> · {timeAgo(doc.upload_timestamp)} · {doc.chunks} chunks</div>
                                  </div>
                                  {doc.doc_type
                                    ? <Pill tone="corrected">{doc.doc_type}</Pill>
                                    : <Pill tone="neutral">Not analyzed</Pill>}
                                </div>
                              ))}
                            </div>
                          )}

                          {totalPages > 1 && (
                            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "12px", marginTop: "14px" }}>
                              <button onClick={() => setAdminDocsPage(p => Math.max(0, p - 1))} disabled={adminDocsPage === 0} style={{
                                background: "none", border: "1px solid #e2e8f0", borderRadius: "8px",
                                padding: "6px 14px", fontSize: "12px", cursor: adminDocsPage === 0 ? "default" : "pointer",
                                color: adminDocsPage === 0 ? "#cbd5e1" : "#374151", fontFamily: UI_FONT, fontWeight: 600,
                              }}>Prev</button>
                              <span style={{ fontSize: "12px", color: "#9ca3af" }}>Page {adminDocsPage + 1} of {totalPages}</span>
                              <button onClick={() => setAdminDocsPage(p => p + 1)} disabled={adminDocsPage + 1 >= totalPages} style={{
                                background: "none", border: "1px solid #e2e8f0", borderRadius: "8px",
                                padding: "6px 14px", fontSize: "12px", cursor: adminDocsPage + 1 >= totalPages ? "default" : "pointer",
                                color: adminDocsPage + 1 >= totalPages ? "#cbd5e1" : "#374151", fontFamily: UI_FONT, fontWeight: 600,
                              }}>Next</button>
                            </div>
                          )}
                        </>
                      )}
                    </>
                  );
                })()}

                {/* ── AUDIT LOG TAB — full searchable admin action history (the Overview
                     "Recent Activity" card only ever showed the last 5). ── */}
                {adminTab === "audit" && (() => {
                  const ACTION_META = {
                    grant_admin:     { text: "granted admin access to", color: "#7c3aed" },
                    revoke_admin:    { text: "revoked admin access from", color: "#9ca3af" },
                    suspend_user:    { text: "suspended", color: "#d97706" },
                    reactivate_user: { text: "reactivated", color: "#16a34a" },
                    delete_user:     { text: "deleted", color: "#dc2626" },
                  };
                  const totalPages = adminAuditFull ? Math.max(1, Math.ceil(adminAuditFull.total_count / ADMIN_PAGE_SIZE)) : 1;
                  return (
                    <>
                      <div style={{ marginBottom: "16px" }}>
                        <div style={{ fontSize: "14px", fontWeight: 700, color: "#111827" }}>Audit Log</div>
                        <div style={{ fontSize: "11px", color: "#9ca3af" }}>{adminAuditFull?.total_count ?? 0} recorded admin actions.</div>
                      </div>

                      {adminAuditLoading && !adminAuditFull && <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "10px" }}>Loading…</div>}

                      {adminAuditFull && (
                        <>
                          <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: "12px", marginBottom: "16px" }}>
                            <ChartCard title="Activity Over Time" subtitle="Daily actions, last 14 days">
                              {adminAuditFull.actions_per_day?.some(d => d.count > 0)
                                ? <ComboBarLine data={adminAuditFull.actions_per_day} height={170} barColor="#7c3aed" />
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No activity in this window.</div>}
                            </ChartCard>
                            <ChartCard title="Actions by Type">
                              {adminAuditFull.action_breakdown?.length
                                ? <div style={{ display: "flex", justifyContent: "center", paddingTop: "6px" }}>
                                    <DonutChart data={adminAuditFull.action_breakdown.map(a => ({ label: (ACTION_META[a.label]?.text || a.label).replace(/^./, c => c.toUpperCase()), value: a.count }))} layout="column" />
                                  </div>
                                : <div style={{ fontSize: "11.5px", color: "#94a3b8" }}>No admin actions recorded yet.</div>}
                            </ChartCard>
                          </div>

                          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "14px" }}>
                            <div style={{ position: "relative", flex: "1 1 200px" }}>
                              <div style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "#9ca3af", display: "flex" }}>
                                <IconSearch width={13} height={13} />
                              </div>
                              <input
                                value={adminAuditSearch}
                                onChange={e => { setAdminAuditSearch(e.target.value); setAdminAuditPage(0); }}
                                placeholder="Search actor or target…"
                                style={{
                                  width: "100%", border: "1px solid #d1d5db", borderRadius: "8px", padding: "7px 10px 7px 30px",
                                  fontSize: "12.5px", fontFamily: UI_FONT, outline: "none", boxSizing: "border-box",
                                }}
                              />
                            </div>
                            <select value={adminAuditActionFilter} onChange={e => { setAdminAuditActionFilter(e.target.value); setAdminAuditPage(0); }} style={{
                              border: "1px solid #d1d5db", borderRadius: "8px", padding: "0 10px",
                              fontSize: "12.5px", fontFamily: UI_FONT, color: "#374151", background: "#fff", cursor: "pointer",
                            }}>
                              <option value="">All actions</option>
                              {Object.keys(ACTION_META).map(a => <option key={a} value={a}>{a.replace(/_/g, " ")}</option>)}
                            </select>
                            <button onClick={() => fetchAuditLogFull(adminAuditPage, adminAuditSearch, adminAuditActionFilter)} title="Refresh" style={{
                              background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px",
                              padding: "0 10px", cursor: "pointer", display: "flex", alignItems: "center", color: "#6b7280",
                            }}><IconRefresh width={13} height={13} /></button>
                          </div>

                          {adminAuditFull.items.length === 0 ? (
                            <div style={{ padding: "40px 20px", background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", textAlign: "center" }}>
                              <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}>No matching actions.</div>
                            </div>
                          ) : (
                            <div style={{ background: "#fff", border: "1px solid #eef0f3", borderRadius: "14px", overflow: "hidden" }}>
                              {adminAuditFull.items.map((entry, i) => {
                                const meta = ACTION_META[entry.action] || { text: entry.action, color: "#6b7280" };
                                return (
                                  <div key={i} style={{
                                    display: "flex", alignItems: "flex-start", gap: "12px", padding: "12px 16px",
                                    borderTop: i === 0 ? "none" : "1px solid #f4f5f7",
                                  }}>
                                    <div style={{
                                      width: "26px", height: "26px", borderRadius: "8px", background: meta.color + "16", color: meta.color,
                                      display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "1px",
                                    }}><IconShield width={13} height={13} /></div>
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                      <div style={{ fontSize: "12.5px", color: "#374151", lineHeight: "1.5" }}>
                                        <strong>{entry.actor}</strong> {meta.text} <strong>{entry.target}</strong>
                                      </div>
                                      {entry.detail && <div style={{ fontSize: "11px", color: "#9ca3af", marginTop: "2px" }}>{entry.detail}</div>}
                                    </div>
                                    <div style={{ fontSize: "10.5px", color: "#c3c2b7", whiteSpace: "nowrap" }}>{timeAgo(entry.timestamp)}</div>
                                  </div>
                                );
                              })}
                            </div>
                          )}

                          {totalPages > 1 && (
                            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "12px", marginTop: "14px" }}>
                              <button onClick={() => setAdminAuditPage(p => Math.max(0, p - 1))} disabled={adminAuditPage === 0} style={{
                                background: "none", border: "1px solid #e2e8f0", borderRadius: "8px",
                                padding: "6px 14px", fontSize: "12px", cursor: adminAuditPage === 0 ? "default" : "pointer",
                                color: adminAuditPage === 0 ? "#cbd5e1" : "#374151", fontFamily: UI_FONT, fontWeight: 600,
                              }}>Prev</button>
                              <span style={{ fontSize: "12px", color: "#9ca3af" }}>Page {adminAuditPage + 1} of {totalPages}</span>
                              <button onClick={() => setAdminAuditPage(p => p + 1)} disabled={adminAuditPage + 1 >= totalPages} style={{
                                background: "none", border: "1px solid #e2e8f0", borderRadius: "8px",
                                padding: "6px 14px", fontSize: "12px", cursor: adminAuditPage + 1 >= totalPages ? "default" : "pointer",
                                color: adminAuditPage + 1 >= totalPages ? "#cbd5e1" : "#374151", fontFamily: UI_FONT, fontWeight: 600,
                              }}>Next</button>
                            </div>
                          )}
                        </>
                      )}
                    </>
                  );
                })()}
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#9ca3af", fontSize: "13px" }}>
              Admin access required.
            </div>
          )
        )}
      </div>

      {/* ── USER DETAIL DRAWER ──────────────────────────────────────────── */}
      {selectedUser && (
        <div style={{ position: "fixed", inset: 0, zIndex: 250, display: "flex", justifyContent: "flex-end" }}>
          <div style={{ position: "absolute", inset: 0, background: "rgba(15,17,20,.35)" }} onClick={() => setSelectedUser(null)} />
          <div style={{
            position: "relative", width: "380px", maxWidth: "90vw", height: "100%",
            background: "#fff", boxShadow: "-8px 0 30px rgba(0,0,0,.12)",
            display: "flex", flexDirection: "column", fontFamily: UI_FONT,
          }}>
            <div style={{ padding: "18px 20px", borderBottom: "1px solid #eef0f3", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <div style={{
                  width: "42px", height: "42px", borderRadius: "50%",
                  background: selectedUser.is_admin ? "linear-gradient(135deg,#a78bfa,#7c3aed)" : "linear-gradient(135deg,#60a5fa,#1d4ed8)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#fff", fontSize: "14px", fontWeight: 700,
                }}>{(selectedUser.username || "?").slice(0, 2).toUpperCase()}</div>
                <div>
                  <div style={{ fontSize: "15px", fontWeight: 700, color: "#111827" }}>{selectedUser.username}</div>
                  <div style={{ fontSize: "11.5px", color: "#9ca3af" }}>{selectedUser.email}</div>
                </div>
              </div>
              <button onClick={() => setSelectedUser(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#9ca3af", display: "flex" }}>
                <IconX width={18} height={18} />
              </button>
            </div>

            <div style={{ padding: "16px 20px", borderBottom: "1px solid #eef0f3", display: "flex", gap: "8px" }}>
              <Pill tone={selectedUser.is_admin ? "admin" : "user"}>{selectedUser.is_admin ? "Admin" : "User"}</Pill>
              <Pill tone={selectedUser.is_active ? "active" : "suspended"}>{selectedUser.is_active ? "Active" : "Suspended"}</Pill>
            </div>

            <div style={{ padding: "16px 20px", borderBottom: "1px solid #eef0f3", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" }}>
              {[
                { label: "Chats", value: selectedUser.chat_count },
                { label: "Documents", value: selectedUser.document_count },
                { label: "Last active", value: selectedUser.last_active ? new Date(selectedUser.last_active).toLocaleDateString() : "Never" },
              ].map(s => (
                <div key={s.label}>
                  <div style={{ fontSize: "9.5px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 700, marginBottom: "3px" }}>{s.label}</div>
                  <div style={{ fontSize: "14px", fontWeight: 700, color: "#111827" }}>{s.value}</div>
                </div>
              ))}
            </div>

            <div style={{ padding: "16px 20px", borderBottom: "1px solid #eef0f3", display: "flex", flexDirection: "column", gap: "8px" }}>
              <button
                onClick={() => updateUser(selectedUser.id, { is_admin: !selectedUser.is_admin })}
                disabled={userActionBusy}
                style={{
                  display: "flex", alignItems: "center", gap: "8px", width: "100%",
                  background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "9px",
                  padding: "9px 12px", cursor: userActionBusy ? "default" : "pointer",
                  fontSize: "12.5px", fontWeight: 600, color: "#374151", fontFamily: UI_FONT,
                }}
              ><IconShield width={14} height={14} />{selectedUser.is_admin ? "Revoke admin access" : "Grant admin access"}</button>

              <button
                onClick={() => setConfirmAction({ type: "suspend", user: selectedUser })}
                disabled={userActionBusy}
                style={{
                  display: "flex", alignItems: "center", gap: "8px", width: "100%",
                  background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "9px",
                  padding: "9px 12px", cursor: userActionBusy ? "default" : "pointer",
                  fontSize: "12.5px", fontWeight: 600, color: selectedUser.is_active ? "#d97706" : "#16a34a", fontFamily: UI_FONT,
                }}
              >{selectedUser.is_active ? <IconBan width={14} height={14} /> : <IconCheckCircle width={14} height={14} />}
                {selectedUser.is_active ? "Suspend account" : "Reactivate account"}</button>

              <button
                onClick={() => setConfirmAction({ type: "delete", user: selectedUser })}
                disabled={userActionBusy}
                style={{
                  display: "flex", alignItems: "center", gap: "8px", width: "100%",
                  background: "#fef2f2", border: "1px solid #fecaca", borderRadius: "9px",
                  padding: "9px 12px", cursor: userActionBusy ? "default" : "pointer",
                  fontSize: "12.5px", fontWeight: 600, color: "#dc2626", fontFamily: UI_FONT,
                }}
              ><IconTrash width={14} height={14} />Delete account</button>
            </div>

            <div style={{ flex: 1, overflow: "auto", padding: "16px 20px" }}>
              <div style={{ fontSize: "10.5px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: ".05em", fontWeight: 700, marginBottom: "10px" }}>Recent activity</div>
              {userActivityLoading && <div style={{ fontSize: "12px", color: "#94a3b8" }}>Loading…</div>}
              {!userActivityLoading && userActivity.length === 0 && (
                <div style={{ fontSize: "12px", color: "#9ca3af" }}>No chats from this user yet.</div>
              )}
              {userActivity.map(chat => (
                <div key={chat.id} style={{ marginBottom: "12px", paddingBottom: "12px", borderBottom: "1px solid #f4f5f7" }}>
                  <div style={{ fontSize: "10px", color: "#c3c2b7", marginBottom: "3px" }}>
                    {chat.timestamp ? new Date(chat.timestamp).toLocaleString() : ""}
                  </div>
                  <div style={{ fontSize: "12px", fontWeight: 600, color: "#1e293b", lineHeight: "1.4" }}>
                    {chat.question?.slice(0, 120)}{chat.question?.length > 120 ? "…" : ""}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── DESTRUCTIVE-ACTION CONFIRM ──────────────────────────────────── */}
      {confirmAction && (
        <ConfirmDialog
          title={confirmAction.type === "delete" ? "Delete this account?" : (confirmAction.user.is_active ? "Suspend this account?" : "Reactivate this account?")}
          message={
            confirmAction.type === "delete"
              ? `${confirmAction.user.username} will be permanently removed. Their past chats and documents are kept for the record but shown as unowned.`
              : confirmAction.user.is_active
                ? `${confirmAction.user.username} will not be able to sign in until reactivated.`
                : `${confirmAction.user.username} will be able to sign in again immediately.`
          }
          confirmLabel={confirmAction.type === "delete" ? "Delete account" : (confirmAction.user.is_active ? "Suspend" : "Reactivate")}
          danger={confirmAction.type === "delete" || confirmAction.user.is_active}
          onCancel={() => setConfirmAction(null)}
          onConfirm={async () => {
            if (confirmAction.type === "delete") {
              await deleteUser(confirmAction.user.id);
            } else {
              await updateUser(confirmAction.user.id, { is_active: !confirmAction.user.is_active });
            }
            setConfirmAction(null);
          }}
        />
      )}

      {/* ── SOURCE PANEL (collapsible) ────────────────────────────────── */}
      {mainView === "chat" && activeDoc && sourceOpen && (
        <div
          ref={sourcePanelRef}
          onMouseUp={handleSourceMouseUp}
          onScrollCapture={() => setSelPopover(null)}
          style={{
            width: "min(46vw, 640px)", flexShrink: 0,
            borderLeft: "1px solid #e5e7eb", position: "relative",
            display: "flex", flexDirection: "column", overflow: "hidden",
          }}>
          <div style={{
            height: "52px", flexShrink: 0, display: "flex", alignItems: "center",
            justifyContent: "space-between", padding: "0 14px", borderBottom: "1px solid #e5e7eb",
            background: "#fff",
          }}>
            <span style={{ fontSize: "12.5px", fontWeight: 600, color: "#374151", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {activeDoc.name}
            </span>
            <button onClick={() => setSourceOpen(false)} title="Close source panel" style={{
              background: "none", border: "none", color: "#6b7280", cursor: "pointer", padding: "4px", display: "flex",
            }}><IconX width={16} height={16} /></button>
          </div>
          <PdfViewer doc={activeDoc} highlightPage={hlPage} />

          {/* Floating "Ask about selection" button — appears right after the
              user releases a text selection anywhere in this panel (real DOM
              text for DOCX/TXT, a transparent pdf.js text layer for PDF). */}
          {selPopover && (
            <button
              onMouseDown={e => e.stopPropagation()}
              onMouseUp={e => e.stopPropagation()}
              onClick={handleAskAboutSelection}
              style={{
                position: "absolute",
                left: selPopover.x, top: Math.max(selPopover.y, 8),
                transform: "translate(-50%, calc(-100% - 8px))",
                zIndex: 30, display: "flex", alignItems: "center", gap: "6px",
                background: "#111827", color: "#fff", border: "none",
                borderRadius: "20px", padding: "7px 14px", fontSize: "12.5px",
                fontWeight: 600, fontFamily: UI_FONT, cursor: "pointer",
                boxShadow: "0 4px 14px rgba(0,0,0,.25)", whiteSpace: "nowrap",
              }}
            >
              <IconScale width={13} height={13} /> Ask about selection
            </button>
          )}
        </div>
      )}

      <Styles />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL STYLES — injected once
// ─────────────────────────────────────────────────────────────────────────────
function Styles() {
  return (
    <style>{`
      @keyframes dot-bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40%            { transform: translateY(-5px); }
      }
      @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: .55; }
      }
      * { box-sizing: border-box; }
      ::-webkit-scrollbar { width: 6px; height: 6px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
      ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
      button:focus { outline: none; }
      textarea:focus { outline: none; }

      /* Invisible text layer laid over each PDF canvas page, purely so its
         real glyphs are selectable (the canvas underneath is just pixels).
         Standard pdf.js text-layer styling — width/height/font-size are set
         inline per span by pdfjs's renderTextLayer via the --scale-factor
         custom property; this just handles positioning and visibility. */
      .pdfTextLayer {
        overflow: hidden;
        line-height: 1;
        opacity: 1;
        -webkit-user-select: text;
        user-select: text;
      }
      .pdfTextLayer span, .pdfTextLayer br {
        color: transparent;
        position: absolute;
        white-space: pre;
        cursor: text;
        transform-origin: 0% 0%;
      }
      .pdfTextLayer ::selection { background: rgba(37,99,235,.35); }
      .pdfTextLayer ::-moz-selection { background: rgba(37,99,235,.35); }
    `}</style>
  );
}
