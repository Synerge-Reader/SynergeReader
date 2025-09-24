import React, { useState, useEffect } from "react";
import FileUpload from "./components/FileUpload";
import TextPreview from "./components/TextPreview";
import AskModal from "./components/AskModal";
import TitleLogo from "./components/TitleLogo";
import Top from "./components/Top";
import Markdown from "react-markdown";
import RatingModal from "./components/RatingModal/RatingModal.jsx";
import UserAuth from "./components/UserAuth/UserAuth.jsx";
import "./GridApp.css";
import Spinner from './components/Spinner/Spinner'
import Notifier from './components/Notifier/Notifier'
const GridApp = () => {
  // Removed 'async' keyword
  const [parsedText, setParsedText] = useState("");
  const [selectedText, setSelectedText] = useState("");
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendMsg, setBackendMsg] = useState("");
  const [askOpen, setAskOpen] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [history, setHistory] = useState([]);
  const [openHistory, setOpenHistory] = useState(false);
  const [model, setModel] = useState("llama3.1:8b");
  const [openRating, setOpenRating] = useState(false);
  const [openAuth, setOpenAuth] = useState(false);
  const [authToken, setAuthToken] = useState('')
  const [notification, setNotification] = useState('')


  useEffect(() => {
    const fetchData = async () => {
      fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/test",
      )
        .then((res) => res.json())
        .then((data) => setBackendMsg(data.message))
        .catch(() => setBackendMsg("Could not connect to backend."));
      const token = localStorage.getItem("authToken"); // get token from localStorage
      if (!token) {
        setHistory([]);
        return;
      }
      else {
        const res = await fetch(
          (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + `/history`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              token,
            }),
          }
        );
        const data = await res.json();
        setHistory(data);
      }
    }
    fetchData();
  }, []);


  const handleFileParsed = (text, name) => {
    setParsedText(text);
    setFileName(name);
    setIsLoading(false);
    setError("");
  };

  const handleAsk = async (question) => {
    if (!selectedText.trim()) {
      setError("Please select some text first.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/ask",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            selected_text: selectedText,
            question,
            model,
            auth_token: authToken
          }),
        },
      );

      if (!res.ok) throw new Error("Backend error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        fullText += decoder.decode(value, { stream: true });
        // optionally: set partial streaming text here
      }

      // Extract entry ID from the end of the response
      let answer = fullText;
      let entryId = null;

      const entryIdMatch = fullText.match(/__ENTRY_ID__(\d+)__/);
      if (entryIdMatch) {
        entryId = parseInt(entryIdMatch[1]);
        // Remove the entry ID marker from the answer
        answer = fullText.replace(/__ENTRY_ID__\d+__/, "").trim();
      }

      console.log("Entry ID:", entryId); // You can use this ID as needed

      setAnswer({
        question,
        answer: answer,
        context_chunks: [],
        relevant_history: [],
        entryId: entryId, // Add the entry ID to your state
      });
    } catch (err) {
      setError("Could not get answer from backend.");
    } finally {
      setIsLoading(false);
      setAskOpen(false);
    }
  };

  const handleTextSelection = (text) => {
    setSelectedText(text);
    if (text.trim()) {
      setAskOpen(true);
    }
  };

  return (
    <>

      <div class="parent">
        {notification && <Notifier message={notification} setNotification={setNotification} />}
        {isLoading && <Spinner />}
        {openRating && (
          <RatingModal setOpenRating={setOpenRating} entryId={answer.entryId} />
        )}
        {openAuth && <UserAuth setOpenAuth={setOpenAuth} setAuthToken={setAuthToken} setNotification={setNotification} />}
        <div class="div4">
          <Top setOpenAuth={setOpenAuth} />
          <hr />
          <TitleLogo></TitleLogo>
          <div className="alpha-subtitle">
            Interactive Human-AI Reading for Any Document: Transforming
            Complexity into Clarity and Insights
          </div>
        </div>

        <div class="div1">
          <div className="doc-section">
            {!fileName && (
              <FileUpload
                onFileParsed={handleFileParsed}
                setIsLoading={setIsLoading}
                setError={setError}
                model={model}
                setModel={setModel}
              />
            )}
            {error && <div className="error-message">{error}</div>}
            {isLoading && <div className="loading-spinner">Processing...</div>}

            {fileName && (
              <TextPreview text={parsedText} onSelect={handleTextSelection} />
            )}
            {fileName && (
              <div className="file-info">
                Uploaded: <span>{fileName}</span>
              </div>
            )}
          </div>
        </div>
        <div class="div2">
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

            <hr />

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
                            <div style={{ fontSize: "0.8em", color: "#888" }}>
                              {h.timestamp}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
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
                      <strong>Selected Text:</strong>{" "}
                      {selectedText.substring(0, 200)}
                      {selectedText.length > 200 ? "..." : ""}
                    </div>
                  )}

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
                      <h3>Response</h3>
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
                                  style={{ marginBottom: 8, fontSize: "0.9em" }}
                                >
                                  {chunk.substring(0, 150)}...
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
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
                                  style={{ marginBottom: 8, fontSize: "0.9em" }}
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
                selectedText={selectedText}
              />
            </div>
          </div>
        </div>

        <div class="div3">
          <footer>
            <hr />
            <div className="footContents">
              <div style={{ color: "#2b926e", fontWeight: 500 }}>
                <p>Status: {backendMsg} </p>
              </div>
              <div>
                <p>
                  © {new Date().getFullYear()} Synergy Reader. All rights
                  reserved.{" "}
                </p>{" "}
              </div>
              <div>
                {" "}
                <p>Report An Issue</p>{" "}
              </div>
            </div>
          </footer>
        </div>
      </div>
    </>
  );
};

export default GridApp;
