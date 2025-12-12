import React, { useState } from "react";
import './CorrectionModal.css';

export default function CorrectionModal({ setOpenCorrection, entryId, originalQuestion, originalAnswer }) {
    const [correctedAnswer, setCorrectedAnswer] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleSubmit = async () => {
        if (!correctedAnswer.trim()) {
            alert("Please provide a corrected answer");
            return;
        }

        setIsSubmitting(true);
        try {
            const res = await fetch(
                (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/submit_correction",
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        chat_id: entryId,
                        corrected_answer: correctedAnswer,
                        comment: "User correction"
                    }),
                }
            );

            if (res.ok) {
                alert("Thank you! Your correction has been saved to the knowledge base.");
                setOpenCorrection(false);
            } else {
                alert("Error saving correction. Please try again.");
            }
        } catch (error) {
            console.error("Error submitting correction:", error);
            alert("Error submitting correction. Please try again.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleMarkCorrect = async () => {
        setIsSubmitting(true);
        try {
            const res = await fetch(
                (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/submit_correction",
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        chat_id: entryId,
                        corrected_answer: originalAnswer, // Confirming original answer is correct
                        comment: "Marked as correct by user"
                    }),
                }
            );

            if (res.ok) {
                alert("Answer marked as correct and saved to knowledge base!");
                setOpenCorrection(false);
            }
        } catch (error) {
            console.error("Error marking as correct:", error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="overlay">
            <div className="modal correction-modal">
                {/* Close button */}
                <button className="close-btn" onClick={() => setOpenCorrection(false)}>
                    ×
                </button>

                <h2 className="modal-title">Answer Feedback</h2>

                {/* Original Q&A Display */}
                <div className="original-qa">
                    <div className="qa-section">
                        <strong>Question:</strong>
                        <div className="qa-content">{originalQuestion}</div>
                    </div>
                    <div className="qa-section">
                        <strong>Original Answer:</strong>
                        <div className="qa-content">{originalAnswer}</div>
                    </div>
                </div>

                {/* Action Buttons */}
                <div className="action-buttons">
                    <button
                        className="correct-btn"
                        onClick={handleMarkCorrect}
                        disabled={isSubmitting}
                    >
                        ✓ Mark as Correct
                    </button>
                    <button
                        className="incorrect-btn"
                        onClick={() => {
                            // Just focus on the correction textarea
                            document.getElementById('corrected-answer')?.focus();
                        }}
                        disabled={isSubmitting}
                        style={{
                            background: '#dc3545',
                            color: 'white',
                            border: 'none',
                            padding: '8px 16px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.9em',
                            marginLeft: '10px'
                        }}
                    >
                        ✗ Mark as Incorrect
                    </button>
                </div>

                <div style={{ textAlign: 'center', margin: '10px 0', color: '#666', fontSize: '0.9em' }}>
                    If incorrect, please provide the correct answer below:
                </div>

                {/* Correction Input */}
                <div className="correction-section">
                    <label htmlFor="corrected-answer">Provide Corrected Answer:</label>
                    <textarea
                        id="corrected-answer"
                        className="correction-box"
                        rows="5"
                        placeholder="Enter the correct answer here..."
                        value={correctedAnswer}
                        onChange={(e) => setCorrectedAnswer(e.target.value)}
                    />
                </div>

                {/* Submit Button */}
                <button
                    className="submit-btn"
                    onClick={handleSubmit}
                    disabled={isSubmitting || !correctedAnswer.trim()}
                >
                    {isSubmitting ? "Submitting..." : "Submit Correction"}
                </button>
            </div>
        </div>
    );
}
