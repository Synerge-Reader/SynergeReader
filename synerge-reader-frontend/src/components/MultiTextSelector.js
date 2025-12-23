import React from "react";
import "./MultiTextSelector.css";

const MultiTextSelector = ({ selections, onRemoveSelection, onClearAll }) => {
    if (!selections || selections.length === 0) {
        return null;
    }

    const totalChars = selections.reduce((sum, sel) => sum + sel.text.length, 0);

    return (
        <div className="multi-selector-container">
            <div className="multi-selector-header">
                <div className="selection-summary">
                    <span className="selection-count-badge">{selections.length}</span>
                    <span className="selection-summary-text">
                        {selections.length} selection{selections.length !== 1 ? 's' : ''}
                        ({totalChars.toLocaleString()} characters)
                    </span>
                </div>
                <button
                    className="clear-all-btn"
                    onClick={onClearAll}
                    title="Clear all selections"
                >
                    Clear All
                </button>
            </div>

            <div className="selections-list">
                {selections.map((selection, index) => (
                    <div key={selection.id} className="selection-item">
                        <div className="selection-badge">{index + 1}</div>
                        <div className="selection-content">
                            <div className="selection-meta">
                                <span className="selection-doc-name" title={selection.documentName}>
                                    📄 {selection.documentName}
                                </span>
                                <span className="selection-char-count">
                                    {selection.text.length} chars
                                </span>
                            </div>
                            <div className="selection-text">
                                {selection.text.substring(0, 150)}
                                {selection.text.length > 150 ? "..." : ""}
                            </div>
                        </div>
                        <button
                            className="selection-remove-btn"
                            onClick={() => onRemoveSelection(selection.id)}
                            title="Remove this selection"
                            aria-label={`Remove selection ${index + 1}`}
                        >
                            ✕
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MultiTextSelector;
