import { useEffect, useRef } from "react";
import styles from "./LandingPage.module.css";

const cx = (...keys) => keys.filter(Boolean).map(k => styles[k]).join(" ");

// Icons below are copied 1:1 from GridApp.jsx's own icon set (same viewBox,
// stroke width, and paths) so the landing page uses exactly the same
// iconography as the app itself, not a different icon family.
const iconBase = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };
const IconScale  = (p) => (<svg {...iconBase} {...p}><path d="M12 3v18M5 21h14" /><path d="M5 7l-3 6a3 3 0 006 0l-3-6zM19 7l-3 6a3 3 0 006 0l-3-6z" /><path d="M5 7h14M12 3L8 7h8l-4-4z" /></svg>);
const IconLayers = (p) => (<svg {...iconBase} {...p}><path d="M12 2l9 5-9 5-9-5 9-5z" /><path d="M3 12l9 5 9-5" /><path d="M3 17l9 5 9-5" /></svg>);
const IconDatabase = (p) => (<svg {...iconBase} {...p}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" /><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" /></svg>);
const IconShield = (p) => (<svg {...iconBase} {...p}><path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-4z" /></svg>);
const IconFile = (p) => (<svg {...iconBase} {...p}><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><path d="M14 2v6h6" /></svg>);
const IconUser = (p) => (<svg {...iconBase} {...p}><circle cx="12" cy="8" r="4" /><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8" /></svg>);
const IconWand = (p) => (<svg {...iconBase} {...p}><path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8L19 13M15 9h0M17.8 6.2L19 5M3 21l9-9M12.2 6.2L11 5" /></svg>);

// Task modes exactly as in GridApp.jsx's TASK_MODES — same labels, same colors.
const TASK_MODES = [
  { label: "Research & Q&A",    color: "#0891b2" },
  { label: "Argument Generator", color: "#7c3aed" },
  { label: "Related Precedents", color: "#7c3aed" },
  { label: "Risk Analysis",      color: "#dc2626" },
];

/**
 * SynergeReader marketing landing page — shown at "/" before entering the
 * app. Calls onEnter() to switch App.js over to the real GridApp. Visual
 * language is pulled directly from the running app (same blue/purple
 * gradients, same #13151a sidebar ink, same card and button conventions,
 * same system font) so this reads as the front door of one product.
 */
export default function LandingPage({ onEnter }) {
  const rootRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.add("js");

    const revealEls = rootRef.current
      ? Array.from(rootRef.current.querySelectorAll("*")).filter((el) => el.classList.contains(styles.r))
      : [];
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add(styles.in); io.unobserve(e.target); }
      });
    }, { threshold: .08, rootMargin: "0px 0px -24px 0px" });
    revealEls.forEach((el) => io.observe(el));

    const revealAll = () => revealEls.forEach((el) => el.classList.add(styles.in));
    const t = setTimeout(revealAll, 2500);
    const onVisible = () => { if (!document.hidden) revealAll(); };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      io.disconnect();
      clearTimeout(t);
      document.removeEventListener("visibilitychange", onVisible);
      document.documentElement.classList.remove("js");
    };
  }, []);

  const handleEnter = (e) => { e.preventDefault(); onEnter(); };

  return (
    <div className={styles.root} ref={rootRef}>
      {/* NAV */}
      <div className={styles.nav}>
        <button className={styles.brand} onClick={(e) => e.preventDefault()}>
          <div className={styles.brandMark}><IconScale /></div>
          <span className={styles.brandName}>SynergeReader</span>
        </button>
        <ul className={styles.navLinks}>
          <li><a href="#problem">Problem</a></li>
          <li><a href="#how">How it works</a></li>
          <li><a href="#features">Features</a></li>
          <li><a href="#origin">Why it exists</a></li>
        </ul>
        <button className={styles.navCta} onClick={handleEnter}>Open the app</button>
      </div>

      {/* HERO */}
      <div className={styles.hero}>
        <div className={styles.heroInner}>
          <div>
            <div className={cx("eyebrow", "r")}>On-premise legal document AI</div>
            <h1 className={cx("heroH1", "r", "d1")}>
              Understand every document — <span className={styles.accent}>without sending a single page out.</span>
            </h1>
            <p className={cx("lede", "r", "d2")} style={{ marginTop: 18 }}>
              SynergeReader reads contracts, statutes, and case files the way an associate would — then lets
              you ask, argue, and extract in plain conversation. Every model runs on your own infrastructure,
              so your documents are never sent to a third-party AI model.
            </p>
            <div className={cx("heroBtns", "r", "d3")}>
              <button className={cx("btn", "btnPrimary")} onClick={handleEnter}>
                <span>Open SynergeReader</span><span className={styles.arrow}>→</span>
              </button>
              <a href="#how" className={cx("btn", "btnGhost")}>
                <span>See how it works</span><span>↓</span>
              </a>
            </div>
          </div>

          {/* a real slice of the product, not an abstract graphic */}
          <div className={cx("mockCard", "r", "d2")}>
            <div className={styles.mockTop}>
              <div className={styles.mockDots}><span></span><span></span><span></span></div>
              <span className={styles.mockTitle}>merger_agreement_v4.pdf</span>
            </div>
            <div className={styles.mockBody}>
              <div className={styles.taskChip}>
                <span className={styles.taskDot} style={{ background: "#0891b2" }} />Research &amp; Q&amp;A
              </div>
              <div className={styles.mockUser}>What's the indemnification cap, and where does it come from?</div>
              <div className={styles.mockAiRow}>
                <div className={styles.mockAvatar}><IconScale /></div>
                <div>
                  <div className={styles.mockAiText}>
                    Under <strong>§8.4</strong>, liability is capped at the escrow amount. §8.3 defines what's covered.
                  </div>
                  <div className={styles.mockCite}>Page 14</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* THREE HIGHLIGHTS */}
        <div className={cx("highlights", "r", "d3")}>
          <div className={styles.highlightsInner}>
            <div className={styles.highlight}>
              <div className={styles.highlightIcon} style={{ background: "linear-gradient(135deg,#3b82f6,#1d4ed8)" }}><IconLayers /></div>
              <div className={styles.highlightTitle}>Ask across the whole matter</div>
              <div className={styles.highlightText}>Query one document or every document you've uploaded, combined. Answers stay grounded — each one traces back to the file it came from.</div>
            </div>
            <div className={styles.highlight}>
              <div className={styles.highlightIcon} style={{ background: "linear-gradient(135deg,#a78bfa,#7c3aed)" }}><IconWand /></div>
              <div className={styles.highlightTitle}>Six tools, not one prompt</div>
              <div className={styles.highlightText}>Argument Generator, Risk Analysis, Clause Extractor, Related Precedents, Summarize, Research &amp; Q&amp;A — each a dedicated view built for that task.</div>
            </div>
            <div className={styles.highlight}>
              <div className={styles.highlightIcon} style={{ background: "linear-gradient(135deg,#34d399,#059669)" }}><IconDatabase /></div>
              <div className={styles.highlightTitle}>A memory that compounds</div>
              <div className={styles.highlightText}>Every verified answer joins a shared knowledge base — semantically searchable, and exportable later as training data for your own model.</div>
            </div>
          </div>
        </div>
      </div>

      {/* PROBLEM */}
      <section id="problem" className={cx("sec", "problemBg")}>
        <div className={styles.secInner}>
          <div className={styles.secHead}>
            <div>
              <div className={cx("eyebrow", "r")}>The case against manual review</div>
              <h2 className={cx("secTitle", "r", "d1")}>Reading isn't the bottleneck. <span className={styles.accent}>Rereading is.</span></h2>
            </div>
            <p className={cx("lede", "r", "d2")}>A merger agreement doesn't get shorter under deadline pressure, and neither does the risk of missing something in page ninety. The tools built to help usually introduce a different problem.</p>
          </div>
          <div className={styles.pGrid}>
            <div className={cx("pCard", "r", "d1")}>
              <div className={styles.pCardMark}>01</div>
              <div className={styles.pCardTitle}>Volume outpaces attention</div>
              <div className={styles.pCardText}>Long contracts get reread, cross-referenced, and reread again. The clause that matters is rarely the one at the top of the file.</div>
            </div>
            <div className={cx("pCard", "r", "d2")}>
              <div className={styles.pCardMark}>02</div>
              <div className={styles.pCardTitle}>Cloud AI risks privilege</div>
              <div className={styles.pCardText}>Pasting client documents into a general chatbot means that text now sits on someone else's servers. For privileged material, that risk is often disqualifying on its own.</div>
            </div>
            <div className={cx("pCard", "r", "d3")}>
              <div className={styles.pCardMark}>03</div>
              <div className={styles.pCardTitle}>Generic models invent things</div>
              <div className={styles.pCardText}>A model with no grounding in your actual documents will confidently answer anyway. Legal work needs answers that cite the page they came from.</div>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how" className={styles.sec}>
        <div className={styles.secInner}>
          <div className={styles.secHead}>
            <div>
              <div className={cx("eyebrow", "r")}>How it works</div>
              <h2 className={cx("secTitle", "r", "d1")}>From upload to <span className={styles.accent}>answer</span>, in one workspace.</h2>
            </div>
            <p className={cx("lede", "r", "d2")}>No training pipeline to configure, no prompt engineering required. Upload, ask, and the right tool does the rest.</p>
          </div>
          <div className={styles.howSteps}>
            <div className={cx("step", "r", "d1")}>
              <div className={cx("stepNum", "tnum")}>01</div>
              <div className={styles.stepTitle}>Upload</div>
              <div className={styles.stepText}>PDF, DOCX, or TXT, one file or many at once. Text is chunked and embedded locally — nothing leaves the machine it's processed on.</div>
            </div>
            <div className={cx("step", "r", "d2")}>
              <div className={cx("stepNum", "tnum")}>02</div>
              <div className={styles.stepTitle}>Ask</div>
              <div className={styles.stepText}>Ask across a single document or everything you've uploaded. Semantic search over pgvector retrieves the passages that actually answer the question.</div>
            </div>
            <div className={cx("step", "r", "d3")}>
              <div className={cx("stepNum", "tnum")}>03</div>
              <div className={styles.stepTitle}>Act</div>
              <div className={styles.stepText}>Switch task modes to build an IRAC argument, flag risk, pull a clause, or summarize — then export the conversation, or feed it back into the knowledge base.</div>
            </div>
          </div>
        </div>
      </section>

      {/* INTERFACE PREVIEW — the one dark section, because the real sidebar is dark */}
      <div className={styles.previewBg}>
        <section className={styles.sec}>
          <div className={styles.secInner}>
            <div className={cx("secHead", "secHeadOnInk")}>
              <div>
                <div className={cx("eyebrow", "onInk", "r")}>The workspace</div>
                <h2 className={cx("secTitle", "secTitleOnInk", "r", "d1")}>Built to work the way<br />a reviewer actually <span className={styles.accent}>works</span>.</h2>
              </div>
              <p className={cx("lede", "ledeOnInk", "r", "d2")}>Document, task tools, and conversation live in one screen — no tab-switching, no exporting a page to ask a question about it.</p>
            </div>
            <div className={cx("frame", "r", "d3")}>
              <div className={styles.frameBar}>
                <div className={styles.frameDots}><span></span><span></span><span></span></div>
                <span className={styles.frameUrl}>merger_agreement_v4.pdf — Research &amp; Q&amp;A</span>
              </div>
              <div className={styles.frameBody}>
                <div className={styles.fpSide}>
                  <div className={styles.fpLabel}>Documents</div>
                  <div className={styles.fpTab}>merger_agreement_v4.pdf</div>
                  <div className={styles.fpTab}>disclosure_schedule.pdf</div>
                  <div className={styles.fpTab}>All Documents (2)</div>
                  <div className={styles.fpLabel}>Task mode</div>
                  {TASK_MODES.map(t => (
                    <div key={t.label} className={cx("fpTab", t.label === "Research & Q&A" && "fpTabActive")}>
                      <span className={styles.fpDot} style={{ background: t.color }} />{t.label}
                    </div>
                  ))}
                </div>
                <div className={styles.fpDoc}>
                  <div className={styles.fpPage}>
                    §8.3 Indemnification Obligations<br />
                    of the Company.<br /><br />
                    <span className={styles.hl}>Subject to the limitations set forth in</span><br />
                    <span className={styles.hl}>this Article VIII, the Company shall</span><br />
                    <span className={styles.hl}>indemnify and hold harmless Parent</span><br />
                    against any and all Losses arising<br />
                    from or related to any breach of any<br />
                    representation or warranty...<br /><br />
                    §8.4 Indemnification Cap.<br /><br />
                    Aggregate liability under §8.3 shall<br />
                    not exceed the amount held in escrow.
                  </div>
                </div>
                <div className={styles.fpChat}>
                  <div className={styles.chatUser}>What's the indemnification cap, and where does it come from?</div>
                  <div className={styles.chatAiRow}>
                    <div className={styles.chatAvatar}><IconScale /></div>
                    <div className={styles.chatAiText}>Under <strong>§8.4</strong>, liability is capped at the escrow amount. §8.3 defines what's covered — breaches of any representation or warranty in the agreement.</div>
                  </div>
                  <div className={styles.chatUser}>Compare that to the disclosure schedule.</div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* FEATURES */}
      <section id="features" className={styles.sec}>
        <div className={styles.secInner}>
          <div className={styles.secHead}>
            <div>
              <div className={cx("eyebrow", "r")}>Features</div>
              <h2 className={cx("secTitle", "r", "d1")}>A complete <span className={styles.accent}>legal workspace.</span></h2>
            </div>
            <p className={cx("lede", "r", "d2")}>Every piece is built for document intelligence specifically — not a general chatbot with a legal-sounding system prompt.</p>
          </div>
          <div className={styles.featGrid}>
            {[
              { icon: IconLayers, grad: "linear-gradient(135deg,#3b82f6,#1d4ed8)", title: "On-premise inference", text: "Llama 3.1 8B and Qwen3 run locally via Ollama, auto-routed by task — heavy reasoning versus a quick lookup — with no model to pick by hand." },
              { icon: IconFile, grad: "linear-gradient(135deg,#22d3ee,#0891b2)", title: "Multi-document reasoning", text: "Select any combination of uploads and ask across all of them. Each answer stays traceable to the document it actually came from." },
              { icon: IconWand, grad: "linear-gradient(135deg,#a78bfa,#7c3aed)", title: "Six dedicated task tools", text: "Argument Generator writes in IRAC. Risk Analysis flags ambiguous language. Clause Extractor, Related Precedents, and Summarize each get their own structured view." },
              { icon: IconDatabase, grad: "linear-gradient(135deg,#34d399,#059669)", title: "A knowledge base that compounds", text: "Verified answers, manual entries, and imported external sources all live in one semantically-indexed base — exportable as JSONL or CSV to fine-tune later." },
              { icon: IconShield, grad: "linear-gradient(135deg,#a78bfa,#7c3aed)", title: "Full administrative oversight", text: "Usage analytics, an audit trail for every account change, and a document-insights view that surfaces facts, keywords, and entities the LLM extracted from each upload." },
              { icon: IconUser, grad: "linear-gradient(135deg,#fbbf24,#d97706)", title: "Built for a firm, not one user", text: "Role-based access separates what admins and members can see: knowledge-base editing, user management, and dashboards all differ by role." },
            ].map((f, i) => (
              <div key={f.title} className={cx("feat", "r", `d${(i % 3) + 1}`)}>
                <div className={styles.featIcon} style={{ background: f.grad }}><f.icon /></div>
                <div className={styles.featTitle}>{f.title}</div>
                <div className={styles.featText}>{f.text}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* TECH STRIP */}
      <div className={styles.tech}>
        <div className={styles.techLabel}>Built with</div>
        <div className={styles.techChips}>
          {["React", "FastAPI", "PostgreSQL", "pgvector", "Ollama", "Llama 3.1 8B", "Qwen3", "Nginx", "Google OAuth"].map(t => (
            <span key={t} className={styles.techChip}>{t}</span>
          ))}
        </div>
      </div>

      {/* ORIGIN */}
      <section id="origin" className={styles.sec}>
        <div className={styles.secInner}>
          <div className={styles.secHead}>
            <div>
              <div className={cx("eyebrow", "r")}>Why it exists</div>
              <h2 className={cx("secTitle", "r", "d1")}>Built around <span className={styles.accent}>one constraint.</span></h2>
            </div>
            <p className={cx("lede", "r", "d2")}>Could a firm get real AI leverage on its documents without ever sending them off-premise? Every architectural choice here starts from that question.</p>
          </div>
          <div className={styles.originLayout}>
            <div>
              <div className={cx("callouts", "r", "d1")}>
                <div className={styles.callout}>
                  <div className={cx("calloutNum", "tnum")}>2</div>
                  <div className={styles.calloutLabel}>Local models,<br />auto-routed by task</div>
                </div>
                <div className={styles.callout}>
                  <div className={styles.calloutNum}>RAG</div>
                  <div className={styles.calloutLabel}>Semantic retrieval<br />via pgvector</div>
                </div>
                <div className={styles.callout}>
                  <div className={cx("calloutNum", "tnum")}>6</div>
                  <div className={styles.calloutLabel}>Purpose-built<br />task tools</div>
                </div>
              </div>
              <p className={cx("lede", "r", "d2")} style={{ marginBottom: 14 }}>Capability and confidentiality are usually a trade-off — the more useful a tool is, the more of your data it wants to see, and the further away it wants to see it from.</p>
              <p className={cx("lede", "r", "d2")}>SynergeReader was built to refuse that trade: process everything locally by default, ground every answer in the actual document, and keep a full record of who did what.</p>
            </div>
            <div className={cx("originCard", "r", "d2")}>
              <div className={styles.originRow}><span className={styles.k}>Inference</span><span className={styles.v}>Local · Ollama</span></div>
              <div className={styles.originRow}><span className={styles.k}>Models</span><span className={styles.v}>Llama 3.1 8B / Qwen3</span></div>
              <div className={styles.originRow}><span className={styles.k}>Retrieval</span><span className={styles.v}>pgvector · semantic</span></div>
              <div className={styles.originRow}><span className={styles.k}>Formats</span><span className={styles.v}>PDF · DOCX · TXT</span></div>
              <div className={styles.originRow}><span className={styles.k}>Access control</span><span className={styles.v}>Role-based</span></div>
              <div className={styles.originRow}><span className={styles.k}>Audit trail</span><span className={styles.v}>Every admin action logged</span></div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <div className={styles.cta}>
        <div className={cx("ctaInner", "r")}>
          <div>
            <h2 className={styles.ctaH}>Bring your first document. See it in under a minute.</h2>
            <p className={styles.ctaSub}>No setup beyond an upload. Ask your first question the moment it finishes processing.</p>
          </div>
          <button className={cx("btn", "btnWhite")} onClick={handleEnter}>
            <span>Open SynergeReader</span><span>→</span>
          </button>
        </div>
      </div>

      {/* FOOTER */}
      <footer className={styles.foot}>
        <button className={styles.footBrand} onClick={(e) => e.preventDefault()}>
          <IconScale width={14} height={14} color="#fff" style={{ opacity: .7 }} />
          <span>SynergeReader</span>
        </button>
        <div className={styles.footCopy}>Local-first legal document intelligence</div>
      </footer>
    </div>
  );
}
