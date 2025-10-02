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
      <div className="modal-content">
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 5 }}>
            <textarea
              id="question"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={4}
              style={{
                width: '100%',
                justifyContent: 'center',
                padding: 12,
                border: '1px solid #ddd',
                borderRadius: 4,
                fontSize: '17px',
                fontFamily: 'inherit'
              }}
              placeholder="Highlight text to ask about it, or type your own question."
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
