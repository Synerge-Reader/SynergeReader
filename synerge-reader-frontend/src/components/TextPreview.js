import React from "react";

const TextPreview = ({ documents = [], onSelect }) => {
  const handleMouseUp = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      onSelect && onSelect(selection.toString());
    }
  };

  if (!documents || documents.length === 0) {
    return (
      <div className="alpha-preview-card" role="region" aria-label="Document preview">
        <div className="alpha-preview-title">
          Document Preview
          <hr></hr>
        </div>
        <div className="alpha-preview-text" style={{ userSelect: 'text' }}>
          No text parsed yet.
        </div>
      </div>
    );
  }

  return (
    <div className="alpha-preview-card" role="region" aria-label="Document previews">
      <div className="alpha-preview-title">
        Document Previews
        <hr></hr>
      </div>
      <div className="alpha-preview-text" style={{ userSelect: 'text' }}>
        {documents.map((doc) => (
          <div key={doc.name} style={{ marginBottom: '16px' }}>
            <h4>{doc.name}</h4>
            <div onMouseUp={handleMouseUp}>
              {doc.text ? doc.text.substring(0, 10000) : "No text parsed yet."}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TextPreview;