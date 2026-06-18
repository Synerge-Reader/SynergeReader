import { useState, useRef, useEffect, useCallback } from "react";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import { GlobalWorkerOptions } from "pdfjs-dist/build/pdf";
import { renderAsync } from "docx-preview";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();

const BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
const TASK_MODES = [
  { id: "research",   label: "Research & Q&A",    model: "llama3.1:8b",  color: "#0891b2" },
  { id: "argument",   label: "Argument Generator", model: "gemma4:31b", color: "#7c3aed" },
  { id: "precedents", label: "Related Precedents", model: "gemma4:31b", color: "#7c3aed" },
  { id: "risk",       label: "Risk Analysis",      model: "gemma4:31b", color: "#dc2626" },
  { id: "clause",     label: "Clause Extractor",   model: "llama3.1:8b",  color: "#0891b2" },
  { id: "summarize",  label: "Summarize",          model: "gemma4:31b", color: "#7c3aed" },
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
// PDF VIEWER — renders each page onto its own <canvas>
// ─────────────────────────────────────────────────────────────────────────────
function PdfCanvasPage({ pdfDoc, pageNum, highlighted }) {
  const canvasRef    = useRef(null);
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
          }}
        />

      </div>

      <style>{`
        .docx-preview-wrapper { padding: 0 !important; background: transparent !important; }
        .docx-preview section.docx {
          width: 100% !important;
          padding: 48px 64px !important;
          box-shadow: none !important;
          margin: 0 !important;
          border-bottom: 1px solid #e5e7eb !important;
          box-sizing: border-box !important;
          background: #fff !important;
        }
        .docx-preview section.docx:last-child { border-bottom: none !important; }
        .docx-preview { font-family: Arial, sans-serif; width: 100% !important; }
        .docx-preview img { max-width: 100%; height: auto; }
      `}</style>
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
  const [rightTab,    setRightTab]    = useState("chat");
  const [navTab,      setNavTab]      = useState("Research");
  const [kbCount,     setKbCount]     = useState(0);
  const [suggestions, setSuggestions] = useState([]);
  const [precedents,  setPrecedents]  = useState([]);
  const [precLoading, setPrecLoading] = useState(false);
  const [kbEntries,   setKbEntries]   = useState([]);
  const [kbLoading,   setKbLoading]   = useState(false);
  const [kbSearch,    setKbSearch]    = useState("");
  const [exportMsg,   setExportMsg]   = useState("");

  const fileInputRef  = useRef(null);
  const chatScrollRef = useRef(null);
  const abortRef      = useRef(null);

  const activeDoc = docs.find(d => d.id === activeDocId) || null;
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
    fetch(`${BACKEND}/knowledge_base/${id}`, { method: "DELETE" })
      .then(() => fetchKbEntries())
      .catch(() => {});
  }, [fetchKbEntries]);

  // CourtListener search — called after document upload
  const searchPrecedents = useCallback(async (docText, docName) => {
    setPrecedents([]);
    setPrecLoading(true);
    try {
      // Build a search query from the doc name and first 200 chars of text
      const query = [
        docName.replace(/\.(pdf|docx|txt)$/i, "").replace(/[_-]/g, " "),
        (docText || "").slice(0, 200),
      ].join(" ").trim().slice(0, 100);

      const res = await fetch(
        `https://www.courtlistener.com/api/rest/v4/search/?q=${encodeURIComponent(query)}&type=o&format=json&page_size=5`,
        { headers: { "Accept": "application/json" } }
      );
      if (!res.ok) throw new Error();
      const data = await res.json();
      const results = (data.results || []).map(r => ({
        name:      r.caseName || r.case_name || "Unknown Case",
        court:     (r.court || r.court_id || "").replace(/_/g, " "),
        date:      (r.dateFiled || r.date_filed || "").slice(0, 4),
        url:       r.absolute_url ? `https://www.courtlistener.com${r.absolute_url}` : null,
        snippet:   r.snippet || "",
        score: null,
      }));
      setPrecedents(results);
    } catch {
      // API unavailable — show empty state, not fake data
      setPrecedents([]);
    } finally {
      setPrecLoading(false);
    }
  }, []);

  const handleCitation = useCallback((page) => {
    setHlPage(page);
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

        setDocs(prev => [...prev, newDoc]);
        setActiveDocId(newDoc.id);
        setHlPage(null);
        setSuggestions([]);
        setMessages([{
          id: Date.now(), role: "assistant", model: task?.model,
          text: `"${file.name}" processed — ${parsed.pages} page${parsed.pages !== 1 ? "s" : ""} indexed. Ask a question or click a suggested question above.`,
          citations: [],
        }]);

        // Generate suggested questions
        fetchSuggestions(parsed.text.slice(0, 2500), file.name, task?.model || "llama3.1:8b");
        searchPrecedents(parsed.text.slice(0, 300), file.name);

      } catch (err) {
        setUploadErr(`Error processing ${file.name}: ${err.message}`);
      }
    }

    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [task]);

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
  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || typing) return;

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const userMsg = { id: Date.now(), role: "user", text };
    setMessages(m => [...m, userMsg]);
    setInput("");
    setTyping(true);

    const prefix = activeTask !== "research" ? TASK_PROMPTS[activeTask] + "\n\n" : "";
    const msgId  = Date.now() + 1;
    setMessages(m => [...m, { id: msgId, role: "assistant", model: task?.model, text: "", citations: [], streaming: true }]);

    try {
      const res = await fetch(`${BACKEND}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          question:             prefix + text,
          model:                task?.model || "llama3.1:8b",
          active_document_name: activeDoc?.name || null,
          selected_text:        "",
          selections:           [],
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
        if (/^__(SEARCHING|READY|CONTEXT|ENTRY_ID)/.test(raw.trim())) continue;
        if (raw.includes("__ENTRY_ID__")) {
          const m = raw.match(/__ENTRY_ID__(\d+)__/);
          if (m) setMessages(prev => prev.map(msg =>
            msg.id === msgId ? { ...msg, entryId: parseInt(m[1]) } : msg
          ));
          continue;
        }
        if (raw.startsWith("__ERROR__")) {
          setMessages(prev => prev.map(msg =>
            msg.id === msgId ? { ...msg, text: "Backend error — check that Ollama is running.", streaming: false } : msg
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
  }, [activeTask, activeDoc, task, typing, handleCitation]);

  // drag-and-drop
  const onDragOver  = e => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = e => { e.preventDefault(); setIsDragging(false); };
  const onDrop      = e => { e.preventDefault(); setIsDragging(false); processFiles(e.dataTransfer.files); };

  const NAV_TABS = ["Research", "Argument Builder", "Knowledge Base", "Case Library", "Export"];

  // ── EMPTY STATE — full-screen upload ─────────────────────────────────────
  if (!docs.length) {
    return (
      <div
        onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
        style={{
          height: "100vh", display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          background: isDragging ? "#eff6ff" : "#f8fafc",
          border: isDragging ? "3px dashed #3b82f6" : "3px dashed transparent",
          fontFamily: "Georgia,'Times New Roman',serif",
          transition: "all .2s",
        }}
      >
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "40px" }}>
          <div style={{
            width: "40px", height: "40px",
            background: "linear-gradient(135deg,#3b82f6,#1d4ed8)",
            borderRadius: "6px", display: "flex", alignItems: "center",
            justifyContent: "center", color: "#fff", fontSize: "22px", fontWeight: 700,
          }}>≡</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: "24px", color: "#0f2340" }}>SynergeReader</div>
            <div style={{ fontSize: "12px", color: "#64748b", marginTop: "1px" }}>Legal Intelligence Platform</div>
          </div>
        </div>

        {/* Upload card */}
        <div style={{
          background: "#fff", border: "1px solid #e2e8f0",
          borderRadius: "6px", padding: "48px 56px",
          textAlign: "center", maxWidth: "500px", width: "90%",
          boxShadow: "0 4px 24px rgba(0,0,0,.07)",
        }}>
          <div style={{ fontSize: "56px", marginBottom: "16px" }}>⚖</div>
          <div style={{ fontWeight: 700, fontSize: "20px", color: "#0f2340", marginBottom: "8px" }}>
            Upload a Legal Document
          </div>
          <div style={{ fontSize: "13px", color: "#64748b", lineHeight: "1.7", marginBottom: "32px" }}>
            Supports PDF, DOCX, and TXT — up to 20 MB<br />
            All processing stays on your server
          </div>

          {uploadErr && (
            <div style={{
              background: "#fee2e2", border: "1px solid #fca5a5",
              borderRadius: "4px", padding: "10px 14px", color: "#991b1b",
              fontSize: "12px", marginBottom: "16px", textAlign: "left",
            }}>{uploadErr}</div>
          )}

          {/* Drop zone */}
          <div style={{
            border: isDragging ? "2px dashed #3b82f6" : "2px dashed #cbd5e1",
            borderRadius: "4px", padding: "20px", marginBottom: "16px",
            background: isDragging ? "#eff6ff" : "#f8fafc",
            fontSize: "12px", color: "#94a3b8", transition: "all .15s",
          }}>
            Drag and drop files here
          </div>

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            style={{
              width: "100%",
              background: uploading ? "#94a3b8" : "#1e3a5f",
              border: "none", color: "#fff",
              padding: "13px 0", borderRadius: "4px",
              cursor: uploading ? "default" : "pointer",
              fontSize: "14px", fontFamily: "Georgia,serif",
              fontWeight: 600, letterSpacing: ".03em",
              transition: "background .15s",
            }}
          >{uploading ? "Processing…" : "Browse Files"}</button>

          <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" multiple
            style={{ display: "none" }} onChange={e => processFiles(e.target.files)} />
        </div>

        <div style={{
          marginTop: "28px", display: "flex", gap: "20px",
          fontSize: "11px", color: "#94a3b8", fontFamily: "'Courier New',monospace",
        }}>
          <span>● Data stays on server</span>
          <span>● Llama 3.1 70B + 8B</span>
          <span>● pgvector retrieval</span>
        </div>

        <Styles />
      </div>
    );
  }

  // ── MAIN 3-PANEL LAYOUT ───────────────────────────────────────────────────
  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100vh",
      fontFamily: "Georgia,'Times New Roman',serif",
      background: "#f1f5f9", color: "#1e293b", fontSize: "13px",
      overflow: "hidden",
    }}>

      {/* ── TOP NAV ──────────────────────────────────────────────────────── */}
      <div style={{
        background: "#0f2340", borderBottom: "2px solid #1e3a5f",
        display: "flex", alignItems: "center",
        height: "46px", padding: "0 16px",
        flexShrink: 0, gap: 0,
      }}>
        {/* logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginRight: "20px", flexShrink: 0 }}>
          <div style={{
            width: "24px", height: "24px",
            background: "linear-gradient(135deg,#3b82f6,#1d4ed8)",
            borderRadius: "3px", display: "flex", alignItems: "center",
            justifyContent: "center", color: "#fff", fontSize: "14px",
          }}>≡</div>
          <span style={{ color: "#f1f5f9", fontWeight: 700, fontSize: "14px" }}>SynergeReader</span>
          <span style={{ color: "#475569", fontSize: "11px", borderLeft: "1px solid #334155", paddingLeft: "8px" }}>
            Legal Intelligence Platform
          </span>
        </div>

        {/* nav tabs */}
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {NAV_TABS.map(tab => (
            <button key={tab} onClick={() => {
                setNavTab(tab);
                // Map nav tabs to right panel tabs
                if (tab === "Knowledge Base") { setRightTab("kb"); fetchKbEntries(); }
                else if (tab === "Case Library") setRightTab("precedents");
                else if (tab === "Argument Builder") { setActiveTask("argument"); setRightTab("chat"); }
                else if (tab === "Export") {
                  // Export chat history as text file
                  if (!messages.length) { alert("No conversation to export yet."); return; }
                  const content = messages.map(m =>
                    `[${m.role === "user" ? "You" : "Assistant"}]\n${m.text}\n`
                  ).join("\n---\n\n");
                  const blob = new Blob([content], { type: "text/plain" });
                  const url  = URL.createObjectURL(blob);
                  const a    = document.createElement("a");
                  a.href = url; a.download = `SynergeReader_Export_${Date.now()}.txt`;
                  a.click(); URL.revokeObjectURL(url);
                  setNavTab("Research");
                }
                else setRightTab("chat");
              }} style={{
              background: "none", border: "none",
              borderBottom: navTab === tab ? "2px solid #3b82f6" : "2px solid transparent",
              color: navTab === tab ? "#93c5fd" : "#64748b",
              padding: "0 14px", height: "46px",
              cursor: "pointer", fontSize: "12px",
              fontFamily: "Georgia,serif", whiteSpace: "nowrap",
              transition: "color .15s",
            }}>{tab}</button>
          ))}
        </div>

        {/* right: model + user */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
          <span style={{ color: "#475569", fontSize: "10px", fontFamily: "'Courier New',monospace" }}>
            {modelName}
          </span>
          <div style={{
            display: "flex", alignItems: "center", gap: "7px",
            background: "#1e3a5f", border: "1px solid #2d4f7c",
            borderRadius: "3px", padding: "4px 10px",
          }}>
            <span style={{ color: "#cbd5e1", fontSize: "11px" }}>SynergeReader</span>
            <div style={{
              width: "22px", height: "22px", background: "#1d4ed8",
              borderRadius: "50%", display: "flex", alignItems: "center",
              justifyContent: "center", color: "#fff", fontSize: "10px", fontWeight: 700,
            }}>SR</div>
          </div>
        </div>
      </div>

      {/* ── BODY ─────────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", minHeight: 0 }}>

        {/* ── LEFT SIDEBAR ─────────────────────────────────────────────── */}
        <div style={{
          width: "210px", flexShrink: 0,
          background: "#fff", borderRight: "1px solid #e2e8f0",
          display: "flex", flexDirection: "column", overflow: "hidden",
        }}>

          {/* upload section */}
          <div style={{ padding: "12px 12px 6px", flexShrink: 0 }}>
            <div style={{
              fontSize: "10px", fontFamily: "'Courier New',monospace",
              color: "#94a3b8", letterSpacing: ".08em",
              textTransform: "uppercase", marginBottom: "8px", fontWeight: 700,
            }}>Documents</div>

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              style={{
                width: "100%",
                background: uploading ? "#94a3b8" : "#1e3a5f",
                border: "none", color: "#f1f5f9",
                padding: "7px 0", borderRadius: "2px",
                cursor: uploading ? "default" : "pointer",
                fontSize: "12px", fontFamily: "Georgia,serif",
                marginBottom: uploadErr ? "4px" : "8px",
                letterSpacing: ".02em", transition: "background .15s",
              }}
            >{uploading ? "Processing…" : "+ Upload Document"}</button>

            <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" multiple
              style={{ display: "none" }} onChange={e => processFiles(e.target.files)} />

            {uploadErr && (
              <div style={{
                fontSize: "10px", color: "#991b1b",
                background: "#fee2e2", border: "1px solid #fca5a5",
                borderRadius: "2px", padding: "4px 7px",
                marginBottom: "6px", lineHeight: "1.4",
              }}>{uploadErr}</div>
            )}
          </div>

          {/* document list */}
          <div style={{ flex: 1, overflow: "auto", padding: "0 8px" }}>
            {docs.map(doc => {
              const active = doc.id === activeDocId;
              return (
                <div key={doc.id}
                  onClick={() => { setActiveDocId(doc.id); setHlPage(null); }}
                  style={{
                    padding: "8px", borderRadius: "2px", marginBottom: "3px",
                    cursor: "pointer",
                    background: active ? "#eff6ff" : "transparent",
                    border: active ? "1px solid #bfdbfe" : "1px solid transparent",
                    transition: "all .12s",
                  }}
                >
                  <div style={{ marginBottom: "3px" }}>
                    <span style={{
                      fontSize: "9px", padding: "1px 5px", borderRadius: "2px",
                      fontFamily: "'Courier New',monospace", fontWeight: 700,
                      textTransform: "uppercase",
                      background: doc.type === "case" ? "#fee2e2" : doc.type === "contract" ? "#dbeafe" : "#f0fdf4",
                      color:       doc.type === "case" ? "#991b1b" : doc.type === "contract" ? "#1d4ed8" : "#166534",
                    }}>{doc.type.slice(0, 3)}</span>
                  </div>
                  <div style={{
                    fontSize: "11px", lineHeight: "1.35",
                    color: active ? "#1d4ed8" : "#374151",
                    fontWeight: active ? 600 : 400,
                    wordBreak: "break-word", marginBottom: "2px",
                  }}>{doc.name}</div>
                  <div style={{ fontSize: "10px", color: "#9ca3af", fontFamily: "'Courier New',monospace" }}>
                    {doc.pages} page{doc.pages !== 1 ? "s" : ""}
                  </div>
                </div>
              );
            })}
          </div>

          {/* task modes */}
          <div style={{ padding: "10px 12px", borderTop: "1px solid #e2e8f0", flexShrink: 0 }}>
            <div style={{
              fontSize: "10px", fontFamily: "'Courier New',monospace",
              color: "#94a3b8", letterSpacing: ".08em",
              textTransform: "uppercase", marginBottom: "7px", fontWeight: 700,
            }}>Task Mode</div>
            {TASK_MODES.map(t => (
              <button key={t.id} onClick={() => setActiveTask(t.id)} style={{
                display: "flex", alignItems: "center", gap: "7px",
                width: "100%", background: "none", border: "none",
                padding: "5px 4px", cursor: "pointer", textAlign: "left",
                borderRadius: "2px",
                color: activeTask === t.id ? "#1d4ed8" : "#4b5563",
                fontSize: "12px", fontFamily: "Georgia,serif",
                fontWeight: activeTask === t.id ? 700 : 400,
                transition: "color .1s",
              }}>
                <span style={{
                  width: "6px", height: "6px", borderRadius: "50%", flexShrink: 0,
                  background: activeTask === t.id ? t.color : "#d1d5db",
                  transition: "background .15s",
                }} />
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── CENTER: PDF VIEWER ────────────────────────────────────────── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
          {/* toolbar */}
          <div style={{
            background: "#fff", borderBottom: "1px solid #e2e8f0",
            padding: "7px 14px", display: "flex",
            alignItems: "center", gap: "8px", flexShrink: 0,
          }}>
            <div style={{
              flex: 1, fontSize: "12px", color: "#374151", fontWeight: 600,
              overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis",
            }}>
              {activeDoc ? activeDoc.name : "No document open"}
            </div>
            {activeDoc && (
              <span style={{
                fontFamily: "'Courier New',monospace", fontSize: "11px",
                color: "#6b7280", background: "#f3f4f6",
                border: "1px solid #e5e7eb", padding: "2px 8px", borderRadius: "2px",
              }}>
                {activeDoc.pages} page{activeDoc.pages !== 1 ? "s" : ""}
              </span>
            )}
            <button onClick={() => fileInputRef.current?.click()} style={{
              background: "#1e3a5f", border: "none", color: "#fff",
              padding: "4px 12px", borderRadius: "2px", cursor: "pointer",
              fontSize: "11px", fontFamily: "Georgia,serif", letterSpacing: ".02em",
            }}>+ Add Doc</button>
          </div>

          <PdfViewer doc={activeDoc} highlightPage={hlPage} />
        </div>

        {/* ── RIGHT PANEL ───────────────────────────────────────────────── */}
        <div style={{
          width: "310px", flexShrink: 0,
          background: "#fff", borderLeft: "1px solid #e2e8f0",
          display: "flex", flexDirection: "column", overflow: "hidden",
        }}>
          {/* panel header */}
          <div style={{ padding: "10px 12px 0", borderBottom: "1px solid #e2e8f0", flexShrink: 0 }}>
            <div style={{
              display: "flex", alignItems: "center",
              justifyContent: "space-between", marginBottom: "8px",
            }}>
              <span style={{ fontWeight: 700, fontSize: "12px", color: "#0f2340", letterSpacing: ".03em" }}>
                Legal Research Assistant
              </span>
              <Badge color={task?.color || "#0891b2"}>{modelName}</Badge>
            </div>
            <div style={{ display: "flex", marginBottom: "-1px" }}>
              {[{ id: "chat", label: "Chat" }, { id: "precedents", label: "Precedents" }, { id: "kb", label: "KB" }].map(rt => (
                <button key={rt.id} onClick={() => setRightTab(rt.id)} style={{
                  background: "none", border: "none",
                  borderBottom: rightTab === rt.id ? "2px solid #1d4ed8" : "2px solid transparent",
                  color: rightTab === rt.id ? "#1d4ed8" : "#64748b",
                  padding: "4px 12px 6px", cursor: "pointer",
                  fontSize: "11px", fontFamily: "Georgia,serif",
                  transition: "color .12s",
                }}>{rt.label}</button>
              ))}
            </div>
          </div>

          {/* ── CHAT ── */}
          {rightTab === "chat" && (
            <>
              {/* suggested questions */}
              {suggestions.length > 0 && (
                <div style={{
                  padding: "8px 10px",
                  borderBottom: "1px solid #f1f5f9",
                  flexShrink: 0,
                }}>
                  <div style={{
                    fontSize: "10px", fontFamily: "'Courier New',monospace",
                    color: "#94a3b8", letterSpacing: ".07em",
                    textTransform: "uppercase", marginBottom: "5px", fontWeight: 700,
                  }}>Suggested Questions</div>
                  {suggestions.map((q, i) => (
                    <button key={i} onClick={() => sendMessage(q)} style={{
                      display: "block", width: "100%",
                      background: "#f8fafc", border: "1px solid #e2e8f0",
                      borderRadius: "2px", padding: "5px 8px",
                      textAlign: "left", fontSize: "11px",
                      color: "#1d4ed8", cursor: "pointer",
                      marginBottom: "4px", fontFamily: "Georgia,serif",
                      lineHeight: "1.35", transition: "background .1s",
                    }}
                      onMouseEnter={e => e.currentTarget.style.background = "#eff6ff"}
                      onMouseLeave={e => e.currentTarget.style.background = "#f8fafc"}
                    >{q}</button>
                  ))}
                </div>
              )}

              {/* messages */}
              <div ref={chatScrollRef} style={{ flex: 1, overflow: "auto", padding: "10px 10px 4px" }}>
                {messages.length === 0 && (
                  <div style={{
                    color: "#94a3b8", fontSize: "12px", textAlign: "center",
                    marginTop: "24px", lineHeight: "1.6",
                    fontFamily: "'Courier New',monospace",
                  }}>
                    Upload a document and ask<br />a legal question to begin.
                  </div>
                )}

                {messages.map(msg => {
                  const isUser = msg.role === "user";
                  return (
                    <div key={msg.id} style={{
                      display: "flex",
                      justifyContent: isUser ? "flex-end" : "flex-start",
                      marginBottom: "12px",
                    }}>
                      <div style={{
                        maxWidth: "95%",
                        background: isUser ? "#1e3a5f" : "#f8fafc",
                        border: isUser ? "none" : "1px solid #e2e8f0",
                        borderRadius: "3px", padding: "10px 12px",
                      }}>
                        {!isUser && msg.model && (
                          <div style={{
                            fontSize: "10px", fontFamily: "'Courier New',monospace",
                            color: task?.color || "#0891b2",
                            fontWeight: 700, marginBottom: "5px",
                            letterSpacing: ".05em", textTransform: "uppercase",
                          }}>
                            {MODEL_LABEL[msg.model] || msg.model}
                          </div>
                        )}
                        <div style={{
                          fontSize: "12px",
                          color: isUser ? "#f1f5f9" : "#1e293b",
                          lineHeight: "1.65", whiteSpace: "pre-wrap",
                        }}>
                          {msg.text}
                          {msg.streaming && (
                            <span style={{ opacity: .5, animation: "blink 1s infinite" }}>▊</span>
                          )}
                        </div>
                        {msg.citations?.length > 0 && (
                          <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "4px" }}>
                            {msg.citations.map((c, i) => (
                              <CitationChip key={i} page={c.page} label="" onClick={handleCitation} />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {typing && <DotsLoader />}
              </div>

              {/* input */}
              <div style={{
                padding: "8px 10px",
                borderTop: "1px solid #e2e8f0",
                display: "flex", gap: "6px", flexShrink: 0,
              }}>
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage(input);
                    }
                  }}
                  placeholder={activeDoc ? "Ask a legal question…" : "Upload a document first…"}
                  disabled={!activeDoc}
                  rows={2}
                  style={{
                    flex: 1, border: "1px solid #d1d5db", borderRadius: "2px",
                    padding: "7px 10px", fontSize: "12px",
                    fontFamily: "Georgia,serif", color: "#1e293b",
                    background: activeDoc ? "#f8fafc" : "#f1f5f9",
                    outline: "none", resize: "none", lineHeight: "1.45",
                  }}
                />
                <button
                  onClick={() => sendMessage(input)}
                  disabled={!activeDoc || !input.trim()}
                  style={{
                    background: (!activeDoc || !input.trim()) ? "#94a3b8" : "#1e3a5f",
                    border: "none", color: "#fff",
                    padding: "7px 14px", borderRadius: "2px",
                    cursor: (!activeDoc || !input.trim()) ? "default" : "pointer",
                    fontSize: "12px", fontFamily: "Georgia,serif",
                    fontWeight: 600, alignSelf: "flex-end",
                    letterSpacing: ".03em", transition: "background .15s",
                  }}
                >Send</button>
              </div>
            </>
          )}

          {/* ── PRECEDENTS ── */}
          {rightTab === "precedents" && (
            <div style={{ flex: 1, overflow: "auto", padding: "10px" }}>
              <div style={{
                fontSize: "10px", fontFamily: "'Courier New',monospace",
                color: "#94a3b8", letterSpacing: ".07em",
                textTransform: "uppercase", marginBottom: "8px", fontWeight: 700,
              }}>Related Precedents — CourtListener</div>

              {precLoading && (
                <div style={{ fontSize: "11px", color: "#94a3b8", fontFamily: "'Courier New',monospace", padding: "8px 0" }}>
                  Searching CourtListener…
                </div>
              )}

              {!precLoading && precedents.length === 0 && (
                <div style={{
                  padding: "12px", background: "#f8fafc",
                  border: "1px solid #e2e8f0", borderRadius: "2px",
                  fontSize: "11px", color: "#64748b", lineHeight: "1.5",
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
                    padding: "10px", background: "#f8fafc",
                    border: "1px solid #e2e8f0", borderRadius: "2px",
                    marginBottom: "6px", cursor: p.url ? "pointer" : "default",
                    transition: "background .1s",
                  }}
                    onClick={() => p.url && window.open(p.url, "_blank")}
                    onMouseEnter={e => e.currentTarget.style.background = "#eff6ff"}
                    onMouseLeave={e => e.currentTarget.style.background = "#f8fafc"}
                  >
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#1d4ed8", marginBottom: "3px" }}>
                      {p.name}
                    </div>
                    <div style={{ fontSize: "10px", color: "#6b7280", fontFamily: "'Courier New',monospace", marginBottom: p.snippet ? "4px" : "6px" }}>
                      {p.court}{p.date ? ` · ${p.date}` : ""}
                      {p.url && <span style={{ color: "#3b82f6", marginLeft: "6px" }}>↗ View</span>}
                    </div>
                    {p.snippet && (
                      <div style={{ fontSize: "11px", color: "#475569", lineHeight: "1.4", marginBottom: "6px", fontStyle: "italic" }}>
                        {p.snippet.slice(0, 120)}…
                      </div>
                    )}
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <div style={{ flex: 1, height: "4px", background: "#e5e7eb", borderRadius: "2px", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${p.score !== null ? p.score + "%" : "N/A"}`, background: col, borderRadius: "2px", transition: "width .6s" }} />
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
                    marginTop: "8px", width: "100%", background: "#f8fafc",
                    border: "1px solid #e2e8f0", borderRadius: "2px",
                    padding: "6px", cursor: "pointer", fontSize: "11px",
                    color: "#374151", fontFamily: "Georgia,serif",
                  }}
                >↺ Search Again</button>
              )}
            </div>
          )}

          {/* ── KB ── */}
          {rightTab === "kb" && (
            <div style={{ flex: 1, overflow: "auto", padding: "10px", display: "flex", flexDirection: "column", gap: "6px" }}>
              {/* header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{
                  fontSize: "10px", fontFamily: "'Courier New',monospace",
                  color: "#94a3b8", letterSpacing: ".07em",
                  textTransform: "uppercase", fontWeight: 700,
                }}>Knowledge Base · {kbCount} entries</div>
                <button onClick={fetchKbEntries} style={{
                  background: "none", border: "none", cursor: "pointer",
                  fontSize: "11px", color: "#3b82f6", fontFamily: "Georgia,serif",
                }}>↺ Refresh</button>
              </div>

              {/* search */}
              <input
                value={kbSearch}
                onChange={e => setKbSearch(e.target.value)}
                placeholder="Search entries…"
                style={{
                  width: "100%", border: "1px solid #d1d5db", borderRadius: "2px",
                  padding: "5px 8px", fontSize: "11px", fontFamily: "Georgia,serif",
                  color: "#1e293b", outline: "none", boxSizing: "border-box",
                }}
              />

              {kbLoading && (
                <div style={{ fontSize: "11px", color: "#94a3b8", fontFamily: "'Courier New',monospace" }}>
                  Loading…
                </div>
              )}

              {!kbLoading && kbEntries.length === 0 && (
                <div style={{
                  padding: "12px", background: "#f8fafc",
                  border: "1px solid #e2e8f0", borderRadius: "2px",
                  fontSize: "11px", color: "#64748b", lineHeight: "1.5",
                }}>
                  No entries yet. KB is populated automatically as you ask questions.
                </div>
              )}

              {kbEntries
                .filter(e => !kbSearch || e.question?.toLowerCase().includes(kbSearch.toLowerCase()) || e.answer?.toLowerCase().includes(kbSearch.toLowerCase()))
                .map((entry, i) => (
                  <div key={entry.id || i} style={{
                    padding: "8px", background: "#f8fafc",
                    border: "1px solid #e2e8f0", borderRadius: "2px",
                  }}>
                    <div style={{ fontSize: "11px", fontWeight: 600, color: "#1e293b", marginBottom: "4px", lineHeight: "1.35" }}>
                      {entry.question?.slice(0, 100)}{entry.question?.length > 100 ? "…" : ""}
                    </div>
                    <div style={{ fontSize: "11px", color: "#475569", lineHeight: "1.4", marginBottom: "5px" }}>
                      {entry.answer?.slice(0, 120)}{entry.answer?.length > 120 ? "…" : ""}
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                        <span style={{ fontSize: "9px", color: "#94a3b8", fontFamily: "'Courier New',monospace" }}>
                          {entry.corrected_by || "auto"} · {entry.usage_count || 0}× used
                        </span>
                      </div>
                      <button
                        onClick={() => deleteKbEntry(entry.id)}
                        style={{
                          background: "none", border: "none", cursor: "pointer",
                          fontSize: "10px", color: "#ef4444", padding: "1px 4px",
                        }}
                        title="Delete entry"
                      >✕</button>
                    </div>
                  </div>
                ))
              }
            </div>
          )}
        </div>
      </div>

      {/* ── STATUS BAR ───────────────────────────────────────────────────── */}
      <div style={{
        background: "#0f2340", borderTop: "1px solid #1e3a5f",
        height: "26px", display: "flex", alignItems: "center",
        padding: "0 16px", gap: "20px", flexShrink: 0,
      }}>
        <StatusDot color="#22c55e" text="All data stays on your server" />
        <StatusDot color={task?.color || "#0891b2"} text={`${modelName} active`} />
        <StatusDot color="#f59e0b" text={`Knowledge base · ${kbCount} entries`} />
        <StatusDot color="#22c55e" text="DGX Server · GPU accelerated" />
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: "10px", color: "#334155", fontFamily: "'Courier New',monospace" }}>
          {task?.label}
        </span>
      </div>

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
      * { box-sizing: border-box; }
      ::-webkit-scrollbar { width: 5px; height: 5px; }
      ::-webkit-scrollbar-track { background: #f1f5f9; }
      ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 2px; }
      ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
      button:focus { outline: none; }
      textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,.15); }
    `}</style>
  );
}
