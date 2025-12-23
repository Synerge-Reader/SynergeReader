import React, { useEffect } from "react";

const TextPreview = ({ documents = [], onSelect, currentDocumentName = null }) => {
  const handleMouseUp = (event, documentName) => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const selectedText = selection.toString().trim();
      
      // Create selection object with metadata
      const selectionObject = {
        id: `sel_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        text: selectedText,
        documentName: documentName,
        timestamp: Date.now(),
      };

      // Always add to selections (multi-select by default)
      // User can clear selections using the Clear All button
      onSelect && onSelect(selectionObject, true);
    }
  };

  // Cleanup URLs when component unmounts or documents change
  useEffect(() => {
    return () => {
      documents.forEach(doc => {
        if (doc.url) {
          URL.revokeObjectURL(doc.url);
        }
      });
    };
  }, [documents]);

  if (!documents || documents.length === 0) {
    return (
      <div className="alpha-preview-card" role="region" aria-label="Document preview">
        <div className="alpha-preview-title">
          Document Preview
          <hr />
        </div>
        <div className="alpha-preview-text" style={{ userSelect: 'text' }}>
          No documents uploaded yet.
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
        <hr />
      </div>
      <div className="alpha-preview-text" style={{ userSelect: 'text' }}>
        {documents.map((doc) => (
          <div key={doc.name} style={{ marginBottom: '24px' }}>
            <h4>{doc.name}</h4>

            {/* Display document in iframe if URL is available */}
            {doc.url && doc.type === 'application/pdf' && (
              <iframe
                title={`viewer-${doc.name}`}
                src={doc.url}
                style={{
                  width: '100%',
                  height: '600px',
                  border: '1px solid #ccc',
                  marginBottom: '16px'
                }}
              />
            )}

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
            <div onMouseUp={(e) => handleMouseUp(e, doc.name)}>
              {doc.text ? doc.text.substring(0, 10000) : "No text parsed yet."}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TextPreview;