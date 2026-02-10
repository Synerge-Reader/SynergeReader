import React, { useState } from "react";

const AskModal = ({ open, onClose, onAsk, selectedText, hasDocuments = true }) => {
  const [question, setQuestion] = useState("");

  if (!open) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question);
      setQuestion("");
    }
  };

  // Determine the mode indicator text
  const getModeIndicator = () => {
    if (selectedText && selectedText.length > 0) {
      return { text: `📝 Context: ${selectedText}`, color: "#2e7d32", bg: "#e8f5e9" };
    }
    if (hasDocuments) {
      return { text: "📄 Using all uploaded documents as context", color: "#1565c0", bg: "#e3f2fd" };
    }
    return { text: "🌐 General chat mode (no documents uploaded)", color: "#7b1fa2", bg: "#f3e5f5" };
  };

  const modeIndicator = getModeIndicator();

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        {/* Mode indicator */}
        <div style={{
          padding: "8px 12px",
          marginBottom: "12px",
          background: modeIndicator.bg,
          color: modeIndicator.color,
          borderRadius: "6px",
          fontSize: "0.85rem",
          fontWeight: "500",
          display: "flex",
          alignItems: "center",
          gap: "6px"
        }}>
          {modeIndicator.text}
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 5 }}>
            <textarea
              id="question"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={4}
              style={{
                width: '90%',
                display: 'block',
                margin: '0 auto',
                padding: 12,
                border: '1px solid #ddd',
                borderRadius: 4,
                fontSize: '17px',
                fontFamily: 'inherit'
              }}
              placeholder={hasDocuments
                ? "Highlight text to ask about it, or type your own question."
                : "Ask any question — I'll answer using my general knowledge."
              }
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

