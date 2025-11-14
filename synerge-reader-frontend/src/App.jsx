import React, { useState, useEffect } from "react";
import FileUpload from "./components/FileUpload";
import TextPreview from "./components/TextPreview";
import AskModal from "./components/AskModal";
import ReactMarkdown from "react-markdown";
import TitleLogo from "./components/TitleLogo";
import Top from "./components/Top";
import "./App.css";

const API_BASE = "/";

export default function App() {
  const [model, setModel] = useState("llama3.1:8b");
  const [documents, setDocuments] = useState([]);
  const [selectedText, setSelectedText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [askOpen, setAskOpen] = useState(false);
  const [backendMsg, setBackendMsg] = useState("");
  const [answer, setAnswer] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}test`)
      .then((res) => res.json())
      .then((data) => setBackendMsg(data.message))
      .catch(() => setBackendMsg("Could not connect to backend."));
    
    // Fetch anonymous user history on load
    fetch(`${API_BASE}history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}), // No token needed for anonymous history
    })
      .then((res) => res.json())
      .then((data) => setHistory(data))
      .catch(() => setHistory([]));
  }, []);

  const handleFileParsed = (text, name) => {
    setDocuments((prevDocs) => {
      const filtered = prevDocs.filter((doc) => doc.name !== name);
      const updated = [...filtered, { name, text }];
      updated.sort((a, b) => a.name.localeCompare(b.name));
      return updated;
    });
    setError("");
  };

  const handleAsk = async (question) => {
    if (!selectedText.trim()) {
      setError("Please select some text from the document before asking a question.");
      return;
    }

    setIsLoading(true);
    setError("");
    setAnswer(null); // Clear previous answer

    try {
      const response = await fetch(`${API_BASE}ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_text: selectedText,
          question,
          model: model,
        }),
      });

      if (!response.ok) {
        throw new Error("Backend error");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullAnswer = "";
      let contextChunks = [];
      let entryId = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        
        if (chunk.includes("__ENTRY_ID__")) {
          const idMatch = chunk.match(/__ENTRY_ID__(\d+)__/);
          if (idMatch && idMatch[1]) {
            entryId = parseInt(idMatch[1], 10);
          }
          continue;
        }

        if (chunk.includes("__ERROR__")) {
          const errorMatch = chunk.match(/__ERROR__(.*)__/);
          if (errorMatch && errorMatch[1]) {
            setError(`Backend streaming error: ${errorMatch[1]}`);
          } else {
            setError("An unknown backend streaming error occurred.");
          }
          continue;
        }

        fullAnswer += chunk;
        setAnswer({
          question,
          answer: fullAnswer,
          context_chunks: contextChunks,
        });
      }

      setIsLoading(false);
      setAskOpen(false);

      // Refresh history
      const historyRes = await fetch(`${API_BASE}history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}), // No token needed for anonymous history
      });
      const historyData = await historyRes.json();
      setHistory(historyData);
    } catch (err) {
      setError("Could not get answer from backend.");
      setIsLoading(false);
    }
  };

  const handleTextSelection = (text) => {
    setSelectedText(text);
    if (text.trim()) {
      setAskOpen(true);
    } else {
      setAskOpen(false);
    }
  };

  return (
    <div className="app-bg">
      <Top />
      <hr />
      <TitleLogo />
      <div className="alpha-subtitle">
        Transforming research papers into interactive AI analysis
      </div>
      <main>
        <FileUpload
          onFileParsed={handleFileParsed}
          setIsLoading={setIsLoading}
          setError={setError}
          model={model}
          setModel={setModel}
        />
        {error && <div className="error-message">{error}</div>}
        {isLoading && <div className="loading-spinner">Processing...</div>}
        {documents.length > 0 && (
          <div className="file-info">
            Uploaded:
            <ul>
              {documents.map((doc) => (
                <li key={doc.name}>{doc.name}</li>
              ))}
            </ul>
          </div>
        )}

        {documents.length > 0 && (
          <TextPreview documents={documents} onSelect={handleTextSelection} />
        )}
        {selectedText && (
          <div
            style={{
              margin: "12px auto",
              maxWidth: 600,
              color: "#3b4ca0",
              background: "#f0f4ff",
              padding: 12,
              borderRadius: 6,
            }}
          >
            <strong>Selected Context:</strong> {selectedText.substring(0, 200)}
            {selectedText.length > 200 ? "..." : ""}
          </div>
        )}
        <AskModal
          open={askOpen}
          onClose={() => setAskOpen(false)}
          onAsk={handleAsk}
          selectedText={selectedText}
          model={model}
        />
        {answer && (
          <div
            style={{
              margin: "32px auto",
              maxWidth: 800,
              background: "#fff",
              borderRadius: 8,
              boxShadow: "0 2px 8px rgba(0,0,0,0.07)",
              padding: 20,
            }}
          >
            <h3>Answer</h3>
            <div style={{ marginBottom: 16 }}>
              <strong>Question:</strong> {answer.question}
            </div>
            <div style={{ marginBottom: 16 }}>
              <strong>Answer:</strong> <ReactMarkdown>{answer.answer}</ReactMarkdown>
            </div>
            {answer.context_chunks && answer.context_chunks.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <strong>Relevant Context:</strong>
                <div
                  style={{
                    background: "#f8f9fa",
                    padding: 12,
                    borderRadius: 4,
                    marginTop: 8,
                  }}
                >
                  {answer.context_chunks.map((chunk, idx) => (
                    <div
                      key={idx}
                      style={{ marginBottom: 8, fontSize: "0.9em" }}
                    >
                      {chunk.substring(0, 150)}...
                    </div>
                  ))}
                </div>
              </div>
            )}
            {answer.relevant_history && answer.relevant_history.length > 0 && (
              <div>
                <strong>Relevant History:</strong>
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
                      style={{ marginBottom: 8, fontSize: "0.9em" }}
                    >
                      <strong>Q:</strong> {hist.question}
                      <br />
                      <strong>A:</strong> {hist.answer.substring(0, 100)}...
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {documents.length > 0 && (
          <div style={{ margin: "32px auto", maxWidth: 800 }}>
            <h3>Chat History</h3>
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
                    <div style={{ fontSize: "0.8em", color: "#888" }}>
                      {h.timestamp}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <footer>
        <hr />
        <div className="footContents">
          <div style={{ color: "#2b926e", fontWeight: 500 }}>
            <p>Status: {backendMsg} </p>
          </div>
          <div>
            <p>© {new Date().getFullYear()} Synergy Reader. All rights reserved. </p>
          </div>
          <div>
            <p>Report An Issue</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
