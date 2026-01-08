import React, { useState, useEffect } from "react";
import FileUpload from "./components/FileUpload";
import TextPreview from "./components/TextPreview";
import AskModal from "./components/AskModal";
import TitleLogo from "./components/TitleLogo";
import Top from "./components/Top";
import RatingModal from "./components/RatingModal/RatingModal.jsx";
import UserAuth from "./components/UserAuth/UserAuth.jsx";
import Spinner from './components/Spinner/Spinner'
import Notifier from './components/Notifier/Notifier'
import Markdown from "react-markdown";
import SurveyModal from "./components/Survey/SurveyModal.jsx";
import CorrectionModal from "./components/CorrectionModal/CorrectionModal.jsx";
import AdminDashboard from "./components/AdminDashboard/AdminDashboard.jsx";
import MultiTextSelector from "./components/MultiTextSelector.js";
import './GridApp.css'
import './App.css'

const GridApp = () => {
  const [parsedDocuments, setParsedDocuments] = useState([]);
  const [selectedTexts, setSelectedTexts] = useState([]); // Changed from selectedText to selectedTexts array
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendMsg, setBackendMsg] = useState("");
  const [askOpen, setAskOpen] = useState(true);
  const [answer, setAnswer] = useState(null);
  const [history, setHistory] = useState([]);
  const [openHistory, setOpenHistory] = useState(false);
  const [model, setModel] = useState("llama3.1:8b");
  const [openRating, setOpenRating] = useState(false);
  const [openAuth, setOpenAuth] = useState(false);
  const [openSurvey, setOpenSurvey] = useState(false);
  const [authToken, setAuthToken] = useState('')
  const [notification, setNotification] = useState('')
  const [openCorrection, setOpenCorrection] = useState(false);
  const [correctionEntry, setCorrectionEntry] = useState(null);
  const [openAdminDashboard, setOpenAdminDashboard] = useState(false);


  useEffect(() => {
    const fetchData = async () => {
      fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/test",
      )
        .then((res) => res.json())
        .then((data) => setBackendMsg(data.message))
        .catch(() => setBackendMsg("Could not connect to backend."));
    }
    fetchData();
    getHistory();
  }, []);

  const getHistory = async () => {
    const token = localStorage.getItem("authToken"); // get token from localStorage

    try {
      const res = await fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + `/history`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            token: token || undefined, // Pass token if available, otherwise undefined for anonymous
          }),
        }
      );
      const data = await res.json();
      setHistory(data);

      // Set auth token if available
      if (token) {
        setAuthToken(token);
      }
    } catch (err) {
      console.error("Error fetching history:", err);
      setHistory([]);
    }
  }




  const handleFileParsed = (doc) => {
    setParsedDocuments(prevDocs => [...prevDocs, doc]);
    setIsLoading(false);
    setError("");
  };

  const handleAsk = async (question) => {
    setIsLoading(true);
    try {
      // If no text is selected, use all parsed documents' text
      let textToSend = "";

      if (selectedTexts && selectedTexts.length > 0) {
        // Concatenate all selections with document context markers
        textToSend = selectedTexts.map((sel, idx) =>
          `[Selection ${idx + 1} from: ${sel.documentName}]\n${sel.text}`
        ).join('\n\n---\n\n');
      } else {
        // Fallback: Concatenate all document texts
        textToSend = parsedDocuments.map(doc => doc.text).join('\n\n---\n\n');
      }

      const res = await fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/ask",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            selected_text: textToSend,
            question,
            model,
            auth_token: authToken
          }),
        },
      );
      console.log(parsedDocuments);
      if (!res.ok) throw new Error("Backend error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullText = "";
      let contextChunks = [];
      let citations = [];
      let apaCitations = [];
      let hasExternalSources = false;
      let citationNote = "No external sources used";
      let similarityScore = 0;
      let contextProcessed = false;

      // Stream and process in real-time
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Handle context metadata
        if (chunk.includes("__CONTEXT__") && !contextProcessed) {
          const contextMatch = chunk.match(/__CONTEXT__({.+?})__/s);
          if (contextMatch) {
            try {
              const contextData = JSON.parse(contextMatch[1]);
              contextChunks = contextData.context_chunks || [];
              citations = contextData.citations || [];
              apaCitations = contextData.apa_citations || [];
              hasExternalSources = contextData.has_external_sources || false;
              citationNote = contextData.citation_note || "No external sources used";
              similarityScore = contextData.similarity_score || 0;
              contextProcessed = true;

              console.log("DEBUG [Frontend]: Context parsed, starting stream display");
              setIsLoading(false);
            } catch (e) {
              console.error("Error parsing context:", e);
            }
          }
          continue;
        }

        // Skip control markers
        if (chunk.includes("__READY__") || chunk.includes("__ENTRY_ID__")) {
          const entryIdMatch = chunk.match(/__ENTRY_ID__(\d+)__/);
          if (entryIdMatch) {
            console.log("Entry ID:", entryIdMatch[1]);
          }
          continue;
        }

        // Skip error markers
        if (chunk.includes("__ERROR__")) {
          continue;
        }

        // Clean up the chunk (remove newlines used as delimiters)
        const cleanChunk = chunk.replace(/\n$/, "");
        if (cleanChunk) {
          fullText += cleanChunk;
          console.log(`DEBUG [Frontend]: Streaming token, fullText length: ${fullText.length}`);

          // Update UI in real-time as we receive tokens
          setAnswer({
            question,
            answer: fullText,
            context_chunks: contextChunks,
            citations: citations,
            apa_citations: apaCitations,
            has_external_sources: hasExternalSources,
            citation_note: citationNote,
            similarity_score: similarityScore,
            relevant_history: [],
          });

          // Small delay to allow React to render
          await new Promise(resolve => setTimeout(resolve, 5));
        }
      }

      // Final update with any remaining data
      let entryId = null;
      const entryIdMatch = fullText.match(/__ENTRY_ID__(\d+)__/);
      if (entryIdMatch) {
        entryId = parseInt(entryIdMatch[1]);
        fullText = fullText.replace(/__ENTRY_ID__\d+__/, "").trim();
      }

      setAnswer({
        question,
        answer: fullText,
        context_chunks: contextChunks,
        citations: citations,
        apa_citations: apaCitations,
        has_external_sources: hasExternalSources,
        citation_note: citationNote,
        similarity_score: similarityScore,
        relevant_history: [],
        entryId: entryId,
      });
    } catch (err) {
      setError("Could not get answer from backend.");
    } finally {
      setIsLoading(false);
      getHistory()
    }
  };

  const handleTextSelection = (selectionObject, isMultiSelect) => {
    if (isMultiSelect) {
      // Append to existing selections
      setSelectedTexts(prev => [...prev, selectionObject]);
    } else {
      // Replace all selections with new one
      setSelectedTexts([selectionObject]);
    }
  };

  const handleRemoveSelection = (selectionId) => {
    setSelectedTexts(prev => prev.filter(sel => sel.id !== selectionId));
  };

  const handleClearAllSelections = () => {
    setSelectedTexts([]);
  };


  return (
    <>
      <div className="parent">
        {notification && <Notifier message={notification} setNotification={setNotification} />}
        {isLoading && <Spinner />}
        {openRating && (
          <RatingModal setOpenRating={setOpenRating} entryId={answer?.entryId} />
        )}
        {openAuth && (
          <UserAuth
            setOpenAuth={setOpenAuth}
            setAuthToken={setAuthToken}
            setNotification={setNotification}
            getHistory={getHistory}
            setOpenSurvey={setOpenSurvey}
          />
        )}
        {openSurvey && (
          <SurveyModal
            setNotification={setNotification}
            setOpenSurvey={setOpenSurvey}


          />
        )}
        {openCorrection && correctionEntry && (
          <CorrectionModal
            setOpenCorrection={setOpenCorrection}
            entryId={correctionEntry.entryId}
            originalQuestion={correctionEntry.question}
            originalAnswer={correctionEntry.answer}
          />
        )}
        {openAdminDashboard && (
          <AdminDashboard
            authToken={authToken}
            onClose={() => setOpenAdminDashboard(false)}
          />
        )}

        {/* Header */}
        <div className="div4">
          <Top
            setOpenAuth={setOpenAuth}
            authToken={authToken}
            setAuthToken={setAuthToken}
            setHistory={setHistory}
            model={model}
            setModel={setModel}
            onAdminClick={() => setOpenAdminDashboard(true)}
          />
          <hr />
          <TitleLogo />
          <div className="alpha-subtitle">
            Interactive Human-AI Reading for Any Document: Transforming
            Complexity into Clarity and Insights
          </div>
        </div>

        {/* Upload / Preview */}
        <div className="div1">
          <div className="doc-section">
            {/* Always show FileUpload - either as main view or as compact upload button */}
            <FileUpload
              onFileParsed={handleFileParsed}
              setIsLoading={setIsLoading}
              setError={setError}
              model={model}
              setModel={setModel}
              isCompact={parsedDocuments.length > 0}
            />

            {error && <div className="error-message">{error}</div>}
            {/*  {isLoading && <div className="loading-spinner">Processing...</div>} */}

            {parsedDocuments.length > 0 && (
              <TextPreview
                documents={parsedDocuments}
                onSelect={handleTextSelection}
              />
            )}
          </div>
        </div>

        {/* Chat + History */}
        <div className="div2">
          <div className="action-box">
            <div className="box-contents">
              <h2
                onClick={() => setOpenHistory(false)}
                style={{
                  cursor: "pointer",
                  fontWeight: openHistory ? 400 : 700,
                  marginRight: 16,
                }}
                aria-selected={!openHistory}
              >
                Chat Box
              </h2>

              <h2
                onClick={() => setOpenHistory(true)}
                style={{
                  cursor: "pointer",
                  fontWeight: openHistory ? 700 : 400,
                }}
              >
                History
              </h2>
            </div>
            <hr></hr>
            <div className="main-action-box">
              {openHistory ? (
                <>
                  <div
                    style={{
                      margin: "32px auto",
                      maxWidth: 800,
                      padding: "10px",
                      marginTop: "-10px",
                    }}
                  >
                    {history.length === 0 ? (
                      <div>No history yet.</div>
                    ) : (
                      <div style={{ maxHeight: 400, overflowY: "auto" }}>
                        {history.map((h, idx) => (
                          <div
                            key={idx}
                            style={{
                              background: "#f8fafc",
                              marginBottom: 12,
                              padding: 16,
                              borderRadius: 8,
                              border: "1px solid #e2e8f0",
                            }}
                          >
                            <div style={{ marginBottom: 8 }}>
                              <strong>Selected Text:</strong>
                              <div
                                style={{
                                  background: "#fff",
                                  padding: 8,
                                  borderRadius: 4,
                                  marginTop: 4,
                                  fontSize: "0.9em",
                                }}
                              >
                                {h.selected_text.substring(0, 200)}
                                {h.selected_text.length > 200 ? "..." : ""}
                              </div>
                            </div>
                            <div style={{ marginBottom: 8 }}>
                              <strong>Q:</strong> {h.question}
                            </div>
                            <div style={{ marginBottom: 8 }}>
                              <strong>A:</strong> {h.answer}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
                  {/* Multi-Text Selector Component */}
                  {selectedTexts && selectedTexts.length > 0 && (
                    <MultiTextSelector
                      selections={selectedTexts}
                      onRemoveSelection={handleRemoveSelection}
                      onClearAll={handleClearAllSelections}
                    />
                  )}

                  {answer && (
                    <div
                      style={{
                        margin: "32px auto",
                        maxWidth: 800,
                        background: "#fff",
                        borderRadius: 8,
                        letterSpacing: "0.5px",
                        padding: 20,
                      }}
                    >
                      <h3>Response </h3>
                      <div style={{ marginBottom: 16 }}>
                        <strong>Question:</strong> {answer.question}
                      </div>
                      <div style={{ marginBottom: 16 }}>
                        <strong>Answer:</strong>
                        <div
                          onClick={() => setOpenRating(true)}
                          style={{
                            cursor: "pointer",
                            display: "inline-block",
                            marginLeft: 8,
                          }}
                        >
                          <Markdown>{answer.answer}</Markdown>
                        </div>
                        <div style={{ marginTop: '10px' }}>
                          <button
                            onClick={() => {
                              setCorrectionEntry({
                                entryId: answer.entryId,
                                question: answer.question,
                                answer: answer.answer
                              });
                              setOpenCorrection(true);
                            }}
                            style={{
                              background: 'none',
                              border: '1px solid #2b926e',
                              color: '#2b926e',
                              padding: '5px 10px',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '0.85em'
                            }}
                          >
                            Provide Feedback / Correct Answer
                          </button>
                        </div>
                      </div>
                      {answer.context_chunks &&
                        answer.context_chunks.length > 0 && (
                          <details style={{ marginBottom: 16 }}>
                            <summary
                              style={{ cursor: "pointer", fontWeight: "bold" }}
                            >
                              Relevant Context
                            </summary>
                            <div
                              style={{
                                background: "#f0f8ff",
                                padding: 12,
                                borderRadius: 4,
                                marginTop: 8,
                              }}
                            >
                              {answer.context_chunks.map((chunk, idx) => (
                                <div
                                  key={idx}
                                  style={{
                                    marginBottom: 8,
                                    fontSize: "0.9em",
                                  }}
                                >
                                  {chunk.substring(0, 150)}...
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      {/* Citations Section - Always show */}
                      <details style={{ marginBottom: 16 }} open>
                        <summary
                          style={{ cursor: "pointer", fontWeight: "bold" }}
                        >
                          Citations (APA Format)
                        </summary>
                        <div
                          style={{
                            background: "#fff8e1",
                            padding: 12,
                            borderRadius: 4,
                            marginTop: 8,
                          }}
                        >
                          {/* Citation Note */}
                          <div
                            style={{
                              marginBottom: 12,
                              padding: "8px 12px",
                              background: answer.has_external_sources ? "#e3f2fd" : "#e8f5e9",
                              borderRadius: 4,
                              fontSize: "0.85em",
                              fontStyle: "italic",
                              color: answer.has_external_sources ? "#1565c0" : "#2e7d32",
                              borderLeft: `3px solid ${answer.has_external_sources ? "#1565c0" : "#2e7d32"}`
                            }}
                          >
                            {answer.citation_note || "No external sources used"}
                          </div>

                          {/* APA Citations */}
                          {answer.apa_citations && answer.apa_citations.length > 0 ? (
                            answer.apa_citations.map((citation, idx) => (
                              <div
                                key={idx}
                                style={{
                                  marginBottom: 12,
                                  fontSize: "0.85em",
                                  color: "#333",
                                  padding: "8px 12px",
                                  background: citation.type === "external" ? "#e3f2fd" : "#fff",
                                  borderRadius: 4,
                                  borderLeft: `3px solid ${citation.type === "external" ? "#2196f3" : "#4caf50"}`
                                }}
                              >
                                <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                                  <span
                                    style={{
                                      fontWeight: "bold",
                                      color: citation.type === "external" ? "#1976d2" : "#388e3c",
                                      minWidth: "fit-content"
                                    }}
                                  >
                                    [{citation.source_num}]
                                  </span>
                                  <span
                                    style={{
                                      fontFamily: "Georgia, serif",
                                      lineHeight: 1.5
                                    }}
                                    dangerouslySetInnerHTML={{
                                      __html: citation.apa
                                        .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                    }}
                                  />
                                </div>
                                <div
                                  style={{
                                    marginTop: 4,
                                    fontSize: "0.8em",
                                    color: "#666"
                                  }}
                                >
                                  {citation.type === "external" ? "📌 External Source" : "📄 Document Source"}
                                </div>
                              </div>
                            ))
                          ) : (
                            <div style={{ color: "#666", fontStyle: "italic" }}>
                              No citations available for this response.
                            </div>
                          )}
                        </div>
                      </details>
                      {answer.relevant_history &&
                        answer.relevant_history.length > 0 && (
                          <details>
                            <summary
                              style={{ cursor: "pointer", fontWeight: "bold" }}
                            >
                              Relevant History
                            </summary>
                            <div
                              style={{
                                background: "#f0f8ff",
                                padding: 12,
                                borderRadius: 4,
                                marginTop: 8,
                              }}
                            >
                              {answer.relevant_history.map((hist, idx) => (
                                <div
                                  key={idx}
                                  style={{
                                    marginBottom: 8,
                                    fontSize: "0.9em",
                                  }}
                                >
                                  <strong>Q:</strong> {hist.question}
                                  <br />
                                  <strong>A:</strong>{" "}
                                  {hist.answer.substring(0, 100)}...
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="main-text-Box">
              <AskModal
                open
                onClose={() => setAskOpen(false)}
                onAsk={handleAsk}
                selectedText={selectedTexts.length > 0 ? `${selectedTexts.length} selection(s)` : ""}
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="div3">
          <footer>
            <hr />
            <div className="footContents">
              <div style={{ color: "#2b926e", fontWeight: 500 }}>
                <p>Status: {backendMsg}</p>
              </div>
              <div>
                <p>
                  © {new Date().getFullYear()} Synergy Reader. All rights
                  reserved.
                </p>
              </div>
              <div>
                <p>Report An Issue</p>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </>
  );
}
export default GridApp;
