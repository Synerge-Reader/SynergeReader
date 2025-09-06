import React from "react";

const TextPreview = ({ text, onSelect }) => {
  const handleMouseUp = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      onSelect && onSelect(selection.toString());
    }
  };
  return (
    <div className="alpha-preview-card" role="region" aria-label="Document preview">
      <div className="alpha-preview-title">Document Preview</div>
      <div className="alpha-preview-text" onMouseUp={handleMouseUp} style={{ userSelect: 'text' }}>
        {text ? text.substring(0, 10000) : "No text parsed yet."}
      </div>
    </div>
  );
};

export default TextPreview;