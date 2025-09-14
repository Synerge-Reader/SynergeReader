import React, { useState } from "react";
import './RatingModal.css'

export default function RatingModal({ setOpenRating, entryId }) {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState("");



  const handleSubmit = async () => {
    const res = await fetch((process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/put_ratings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rating: rating,
        comment: comment,
        id: entryId
      }),
    });
    console.log(rating, comment);
    setOpenRating(false);
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

