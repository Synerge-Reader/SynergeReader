import React, { useEffect, useState } from "react";
import { Viewer } from '@react-pdf-viewer/core';
import { zoomPlugin } from '@react-pdf-viewer/zoom';
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/zoom/lib/styles/index.css';

const TextPreview = ({ documents = [], onSelect, currentDocumentName = null }) => {
  const zoomPluginRef = React.useRef(zoomPlugin());
  const [activeTabIndex, setActiveTabIndex] = useState(0);

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

  // Cleanup URLs only when component unmounts
  useEffect(() => {
    return () => {
      documents.forEach(doc => {
        if (doc.url) {
          URL.revokeObjectURL(doc.url);
        }
      });
    };
  }, []);

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
        <hr />
      </div>
      
      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        borderBottom: '2px solid #ddd',
        overflowX: 'auto',
        backgroundColor: '#f9f9f9'
      }}>
        {documents.map((doc, index) => (
          <button
            key={doc.name}
            onClick={() => setActiveTabIndex(index)}
            style={{
              padding: '12px 16px',
              border: 'none',
              backgroundColor: activeTabIndex === index ? '#fff' : '#f9f9f9',
              borderBottom: activeTabIndex === index ? '3px solid #007bff' : 'none',
              color: activeTabIndex === index ? '#007bff' : '#666',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: activeTabIndex === index ? '600' : '400',
              whiteSpace: 'nowrap',
              transition: 'all 0.2s ease'
            }}
          >
            {doc.name}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="alpha-preview-text" style={{ userSelect: 'text' }}>
        {documents[activeTabIndex] && (
          <div style={{ marginBottom: '24px' }}>
            {/* Display PDF with React PDF Viewer if URL is available */}
            {documents[activeTabIndex].url && documents[activeTabIndex].type === 'application/pdf' && (
              <div
                style={{
                  width: '100%',
                  height: '800px',
                  border: '1px solid #ccc',
                  marginBottom: '16px',
                  display: 'flex',
                  flexDirection: 'column'
                }}
                onMouseUp={(e) => handleMouseUp(e, documents[activeTabIndex].name)}
              >
                <Viewer
                  fileUrl={documents[activeTabIndex].url}
                  plugins={[zoomPluginRef.current]}
                  defaultScale={1.2}
                />
              </div>
            )}

            {/* Display raw text for .txt files */}
            {documents[activeTabIndex].type === 'text/plain' && (
              <div
                style={{
                  width: '100%',
                  height: '800px',
                  border: '1px solid #ccc',
                  marginBottom: '16px',
                  padding: '16px',
                  overflowY: 'auto',
                  backgroundColor: '#f5f5f5',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  fontSize: '14px',
                  lineHeight: '1.5'
                }}
                onMouseUp={(e) => handleMouseUp(e, documents[activeTabIndex].name)}
              >
                {documents[activeTabIndex].text || 'No text content available.'}
              </div>
            )}

            {documents[activeTabIndex].citation && (documents[activeTabIndex].citation.title || documents[activeTabIndex].citation.author) && (
              <div className="citation-info" style={{
                fontSize: '0.85em',
                color: '#555',
                background: '#f5f5f5',
                padding: '8px',
                borderRadius: '4px',
                marginBottom: '8px'
              }}>
                <strong>Citation: </strong>
                {documents[activeTabIndex].citation.title && <span>"{documents[activeTabIndex].citation.title}" </span>}
                {documents[activeTabIndex].citation.author && <span>by {documents[activeTabIndex].citation.author} </span>}
                {documents[activeTabIndex].citation.publication_date && <span>({documents[activeTabIndex].citation.publication_date}) </span>}
                {documents[activeTabIndex].citation.source && <span>- {documents[activeTabIndex].citation.source} </span>}
                {documents[activeTabIndex].citation.doi_url && <span>[{documents[activeTabIndex].citation.doi_url}]</span>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TextPreview;
