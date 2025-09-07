import React, { useState } from "react";

const AskModal = ({ open, onClose, onAsk, selectedText }) => {
  const [question, setQuestion] = useState("");

  if (!open) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question);
      setQuestion("");
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{ background: '#fff', padding: 24, borderRadius: 8, maxWidth: 600, margin: '40px auto', maxHeight: '80vh', overflowY: 'auto', boxShadow: '0 2px 16px rgba(0,0,0,0.15)' }}>
        <h2>Ask a Question</h2>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label htmlFor="question" style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
              Your Question:
            </label>
            <textarea
              id="question"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={4}
              style={{
                width: '100%',
                padding: 12,
                border: '1px solid #ddd',
                borderRadius: 4,
                fontSize: '14px',
                fontFamily: 'inherit'
              }}
              placeholder="Type your question about the selected text..."
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '8px 16px',
                border: '1px solid #ddd',
                borderRadius: 4,
                background: '#fff',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!question.trim()}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: 4,
                background: question.trim() ? '#007bff' : '#ccc',
                color: '#fff',
                cursor: question.trim() ? 'pointer' : 'not-allowed'
              }}
            >
              Ask Question
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AskModal;
