import React, { useState, useEffect } from "react";
import "./AdminDashboard.css";

export default function AdminDashboard({ authToken, onClose }) {
  const [ratings, setRatings] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    fetchRatingsAndStats();
  }, []);

  const fetchRatingsAndStats = async () => {
    setLoading(true);
    setError("");
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
      
      const [ratingsRes, statsRes] = await Promise.all([
        fetch(`${backendUrl}/admin/ratings?token=${authToken}`),
        fetch(`${backendUrl}/admin/ratings/stats?token=${authToken}`)
      ]);

      if (!ratingsRes.ok || !statsRes.ok) {
        throw new Error("Failed to fetch data");
      }

      const ratingsData = await ratingsRes.json();
      const statsData = await statsRes.json();

      setRatings(ratingsData.ratings || []);
      setStats(statsData);
    } catch (err) {
      setError("Failed to load admin dashboard data");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getStarColor = (rating) => {
    if (rating <= 2) return "#e74c3c";
    if (rating <= 3) return "#f39c12";
    if (rating <= 4) return "#f1c40f";
    return "#2ecc71";
  };

  const filteredRatings = ratings.filter((r) => {
    if (filter === "all") return true;
    return r.rating === parseInt(filter);
  });

  if (loading) {
    return (
      <div className="admin-overlay">
        <div className="admin-modal">
          <button className="close-btn" onClick={onClose}>×</button>
          <div style={{ textAlign: "center", padding: "40px" }}>
            <p>Loading admin dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="admin-overlay">
      <div className="admin-modal">
        <button className="close-btn" onClick={onClose}>×</button>
        
        <h2 className="admin-title">Admin Dashboard - Response Ratings</h2>
        
        {error && <div className="admin-error">{error}</div>}

        {stats && (
          <div className="stats-container">
            <div className="stat-card">
              <div className="stat-label">Total Ratings</div>
              <div className="stat-value">{stats.total_ratings}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Average Rating</div>
              <div className="stat-value">{stats.average_rating}/5</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Distribution</div>
              <div className="distribution-mini">
                {[1, 2, 3, 4, 5].map((star) => (
                  <div key={star} className="dist-bar" style={{ "--count": stats.distribution[star] || 0, "--max": Math.max(...Object.values(stats.distribution), 1) }}>
                    {star}★
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="filter-section">
          <label htmlFor="rating-filter">Filter by Rating:</label>
          <select
            id="rating-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Ratings</option>
            {[1, 2, 3, 4, 5].map((star) => (
              <option key={star} value={star}>
                {star} Star{star !== 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </div>

        <div className="ratings-table">
          <div className="table-header">
            <div className="col-user">User</div>
            <div className="col-date">Date</div>
            <div className="col-rating">Rating</div>
            <div className="col-question">Question</div>
            <div className="col-comment">Feedback</div>
          </div>

          <div className="table-body">
            {filteredRatings.length === 0 ? (
              <div className="no-ratings">
                <p>No ratings found</p>
              </div>
            ) : (
              filteredRatings.map((rating) => (
                <div key={rating.id} className="table-row">
                  <div className="col-user">{rating.username}</div>
                  <div className="col-date">
                    {new Date(rating.timestamp).toLocaleDateString()}
                  </div>
                  <div className="col-rating">
                    <div className="stars" style={{ color: getStarColor(rating.rating) }}>
                      {"★".repeat(rating.rating)}
                      {"☆".repeat(5 - rating.rating)}
                    </div>
                  </div>
                  <div className="col-question">
                    <div className="question-text">{rating.question}</div>
                    <div className="selected-text">
                      Context: {rating.selected_text?.substring(0, 80)}...
                    </div>
                  </div>
                  <div className="col-comment">
                    {rating.comment || <span className="no-comment">No feedback</span>}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="admin-footer">
          <p>Showing {filteredRatings.length} of {ratings.length} ratings</p>
          <button className="refresh-btn" onClick={fetchRatingsAndStats}>
            Refresh Data
          </button>
        </div>
      </div>
    </div>
  );
}
