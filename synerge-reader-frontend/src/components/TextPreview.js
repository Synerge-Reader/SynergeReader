import React, { useEffect, useState } from "react";
import { Viewer } from '@react-pdf-viewer/core';
import { zoomPlugin } from '@react-pdf-viewer/zoom';
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/zoom/lib/styles/index.css';

const TextPreview = ({ documents = [], onSelect, currentDocumentName = null, onDeleteDocument }) => {
  const zoomPluginRef = React.useRef(zoomPlugin());
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [deletingDoc, setDeletingDoc] = useState(null);

  // Handle delete document
  const handleDeleteDocument = async (e, docName, index) => {
    e.stopPropagation(); // Prevent tab switching

    if (deletingDoc === docName) return; // Already deleting

    setDeletingDoc(docName);

    // Call parent's delete handler
    if (onDeleteDocument) {
      await onDeleteDocument(docName, index);
    }

    // Adjust active tab if needed
    if (index === activeTabIndex) {
      // If deleting active tab, switch to previous or next
      if (index > 0) {
        setActiveTabIndex(index - 1);
      } else if (documents.length > 1) {
        setActiveTabIndex(0);
      }
    } else if (index < activeTabIndex) {
      // If deleting a tab before active, adjust index
      setActiveTabIndex(prev => prev - 1);
    }

    setDeletingDoc(null);
  };

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
          <div
            key={doc.name}
            style={{
              display: 'flex',
              alignItems: 'center',
              position: 'relative',
            }}
          >
            <button
              onClick={() => setActiveTabIndex(index)}
              style={{
                padding: '12px 16px',
                paddingRight: onDeleteDocument ? '32px' : '16px',
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
            {/* Delete button for each document */}
            {onDeleteDocument && (
              <button
                onClick={(e) => handleDeleteDocument(e, doc.name, index)}
                disabled={deletingDoc === doc.name}
                style={{
                  position: 'absolute',
                  right: '4px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '20px',
                  height: '20px',
                  padding: 0,
                  border: 'none',
                  borderRadius: '50%',
                  backgroundColor: deletingDoc === doc.name ? '#ccc' : 'transparent',
                  color: '#999',
                  cursor: deletingDoc === doc.name ? 'wait' : 'pointer',
                  fontSize: '14px',
                  fontWeight: 'bold',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.15s ease',
                  opacity: 0.6,
                }}
                onMouseEnter={(e) => {
                  if (deletingDoc !== doc.name) {
                    e.currentTarget.style.backgroundColor = '#ff4444';
                    e.currentTarget.style.color = '#fff';
                    e.currentTarget.style.opacity = '1';
                  }
                }}
                onMouseLeave={(e) => {
                  if (deletingDoc !== doc.name) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = '#999';
                    e.currentTarget.style.opacity = '0.6';
                  }
                }}
                title={`Remove ${doc.name}`}
                aria-label={`Remove document ${doc.name}`}
              >
                {deletingDoc === doc.name ? '...' : '×'}
              </button>
            )}
          </div>
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
                  defaultScale={1}
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
