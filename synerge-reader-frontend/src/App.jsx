import React, { useState, useEffect } from "react";
import FileUpload from "./components/FileUpload";
import TextPreview from "./components/TextPreview";
import AskModal from "./components/AskModal";
import TitleLogo from "./components/TitleLogo";
import Top from "./components/Top";
// import axios from "axios";
import "./App.css";

function App() {
  const [parsedText, setParsedText] = useState("");
  const [selectedText, setSelectedText] = useState("");
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendMsg, setBackendMsg] = useState("");
  const [askOpen, setAskOpen] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch(process.env.REACT_APP_BACKEND_URL || "http://localhost:5000" + "/test")
      .then((res) => res.json())
      .then((data) => setBackendMsg(data.message))
      .catch(() => setBackendMsg("Could not connect to backend."));
    fetch(process.env.REACT_APP_BACKEND_URL || "http://localhost:5000" + "/history")
      .then((res) => res.json())
      .then((data) => setHistory(data))
      .catch(() => setHistory([]));
  }, []);

  const handleFileParsed = (text, name) => {
    setParsedText(text);
    setFileName(name);
    setIsLoading(false);
    setError("");
  };

  const handleAsk = (question) => {
    if (!selectedText.trim()) {
      setError("Please select some text from the document before asking a question.");
      return;
    }

    setIsLoading(true);
    fetch(process.env.REACT_APP_BACKEND_URL || "http://localhost:5000" + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        selected_text: selectedText,
        question: question 
      })
    })
      .then(res => res.json())
      .then(data => {
        setAnswer(data);
        setIsLoading(false);
        setAskOpen(false);
        // Refresh history
        fetch(process.env.REACT_APP_BACKEND_URL || "http://localhost:5000" + "/history")
          .then((res) => res.json())
          .then((data) => setHistory(data));
      })
      .catch(() => {
        setError("Could not get answer from backend.");
        setIsLoading(false);
      });
  };

  const handleTextSelection = (text) => {
    setSelectedText(text);
    if (text.trim()) {
      setAskOpen(true);
    }
  };

  return (
    <div className="app-bg"> 
    <Top></Top>
    <hr/>
        <TitleLogo></TitleLogo>
          <div className="alpha-subtitle">
          Transforming research papers into interactive  AI analysis
        </div>
      <main>
        <FileUpload
          onFileParsed={handleFileParsed}
          setIsLoading={setIsLoading}
          setError={setError}
        />
        {error && <div className="error-message">{error}</div>}
        {isLoading && <div className="loading-spinner">Processing...</div>}
        {fileName && (
          <div className="file-info">
            Uploaded: <span>{fileName}</span>
          </div>
        )}
        {fileName &&(
        <TextPreview text={parsedText} onSelect={handleTextSelection} />
)}
        {selectedText && (
          <div style={{margin: '12px auto', maxWidth: 600, color: '#3b4ca0', background: '#f0f4ff', padding: 12, borderRadius: 6}}>
            <strong>Selected Context:</strong> {selectedText.substring(0, 200)}{selectedText.length > 200 ? '...' : ''}
          </div>
        )}
        <AskModal 
          open={askOpen} 
          onClose={() => setAskOpen(false)} 
          onAsk={handleAsk}
          selectedText={selectedText}
        />
        {answer && (
          <div style={{margin: '32px auto', maxWidth: 800, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,0.07)', padding: 20}}>
            <h3>Answer</h3>
            <div style={{marginBottom: 16}}>
              <strong>Question:</strong> {answer.question}
            </div>
            <div style={{marginBottom: 16}}>
              <strong>Answer:</strong> {answer.answer}
            </div>
            {answer.context_chunks && answer.context_chunks.length > 0 && (
              <div style={{marginBottom: 16}}>
                <strong>Relevant Context:</strong>
                <div style={{background: '#f8f9fa', padding: 12, borderRadius: 4, marginTop: 8}}>
                  {answer.context_chunks.map((chunk, idx) => (
                    <div key={idx} style={{marginBottom: 8, fontSize: '0.9em'}}>
                      {chunk.substring(0, 150)}...
                    </div>
                  ))}
                </div>
              </div>
            )}
            {answer.relevant_history && answer.relevant_history.length > 0 && (
              <div>
                <strong>Relevant History:</strong>
                <div style={{background: '#f0f8ff', padding: 12, borderRadius: 4, marginTop: 8}}>
                  {answer.relevant_history.map((hist, idx) => (
                    <div key={idx} style={{marginBottom: 8, fontSize: '0.9em'}}>
                      <strong>Q:</strong> {hist.question}<br/>
                      <strong>A:</strong> {hist.answer.substring(0, 100)}...
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {fileName && (
        <div style={{margin: '32px auto', maxWidth: 800}}>
          <h3>Chat History</h3>
          {history.length === 0 ? <div>No history yet.</div> : (
            <div style={{maxHeight: 400, overflowY: 'auto'}}>
              {history.map((h, idx) => (
                <div key={idx} style={{background: '#f8fafc', marginBottom: 12, padding: 16, borderRadius: 8, border: '1px solid #e2e8f0'}}>
                  <div style={{marginBottom: 8}}>
                    <strong>Selected Text:</strong> 
                    <div style={{background: '#fff', padding: 8, borderRadius: 4, marginTop: 4, fontSize: '0.9em'}}>
                      {h.selected_text.substring(0, 200)}{h.selected_text.length > 200 ? '...' : ''}
                    </div>
                  </div>
                  <div style={{marginBottom: 8}}>
                    <strong>Q:</strong> {h.question}
                  </div>
                  <div style={{marginBottom: 8}}>
                    <strong>A:</strong> {h.answer}
                  </div>
                  <div style={{fontSize: '0.8em', color: '#888'}}>{h.timestamp}</div>
                </div>
              ))}
            </div>
          )}
        </div>
        )}
             
      </main>
    
        
      <footer>  
  <hr/>     
 <div className="footContents">
  <div style={{color: "#2b926e", fontWeight: 500 }}><p>Status: {backendMsg} </p></div>
   <div><p>© {new Date().getFullYear()} Synergy Reader. All rights reserved. </p> </div>
   <div> <p>Report An Issue</p>  </div>
   </div>
</footer>
    </div>
  );
}

export default App;