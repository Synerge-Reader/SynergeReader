import React, { useState } from "react";
import './RatingModal.css'

export default function RatingModal({ setOpenRating, entryId, authToken }) {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState("");



  // /put_ratings is a PUT that accepts a rating only from the signed-in owner
  // of the answer, identified by the Authorization header.
  const handleSubmit = async () => {
    try {
      const res = await fetch((process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/put_ratings", {
        method: "PUT",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken || ""}` },
        body: JSON.stringify({
          rating: rating,
          comment: comment,
          id: entryId
        }),
      });
      if (!res.ok) {
        alert(res.status === 401 ? "Please sign in again to rate this answer." : "Could not save rating.");
        return;
      }
      setOpenRating(false);
    } catch (error) {
      console.error("Error submitting rating:", error);
      alert("Could not save rating.");
    }
  }


  return (
    <div className="overlay">
      <div className="modal">
        {/* Close button */}
        <button className="close-btn" onClick={() => setOpenRating(false)}>
          ×
        </button>

        <h2 className="modal-title">Rate Your Experience</h2>

        {/* Star Rating */}
        <div className="stars">
          {[1, 2, 3, 4, 5].map((star) => (
            <span
              key={star}
              className={`star ${star <= (hover || rating) ? "active" : ""}`}
              onClick={() => setRating(star)}
              onMouseEnter={() => setHover(star)}
              onMouseLeave={() => setHover(0)}
            >
              ★
            </span>
          ))}
        </div>

        {/* Comment Box */}
        <textarea
          className="comment-box"
          rows="3"
          placeholder="Leave a comment..."
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />

        {/* Submit Button */}
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={!rating}
        >
          Submit
        </button>
      </div>
    </div>
  );
}

