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
          {documents.length > 0 && (
            <div className="file-info">
              Uploaded:{" "}
              <span>{documents.map((d) => d.name).join(", ")}</span>
            </div>
          )}
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
        {documents.length > 0 && (
          <div className="file-info">
            Uploaded:{" "}
            <span>{documents.map((d) => d.name).join(", ")}</span>
          </div>
        )}
        <hr></hr>
      </div>
      <div className="alpha-preview-text" style={{ userSelect: 'text' }}>
        {documents.map((doc) => (
          <div key={doc.name} style={{ marginBottom: '16px' }}>
            <h4>{doc.name}</h4>
            {doc.citation && (doc.citation.title || doc.citation.author) && (
              <div className="citation-info" style={{
                fontSize: '0.85em',
                color: '#555',
                background: '#f5f5f5',
                padding: '8px',
                borderRadius: '4px',
                marginBottom: '8px'
              }}>
                <strong>Citation: </strong>
                {doc.citation.title && <span>"{doc.citation.title}" </span>}
                {doc.citation.author && <span>by {doc.citation.author} </span>}
                {doc.citation.publication_date && <span>({doc.citation.publication_date}) </span>}
                {doc.citation.source && <span>- {doc.citation.source} </span>}
                {doc.citation.doi_url && <span>[{doc.citation.doi_url}]</span>}
              </div>
            )}
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