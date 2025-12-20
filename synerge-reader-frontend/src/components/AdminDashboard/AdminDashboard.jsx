import React, { useState, useEffect } from "react";
import "./AdminDashboard.css";

export default function AdminDashboard({ authToken, onClose }) {
  const [ratings, setRatings] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [activeTab, setActiveTab] = useState("ratings");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [trainingFile, setTrainingFile] = useState(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState("");
  const [trainingError, setTrainingError] = useState("");
  const [modelFamilies, setModelFamilies] = useState([]);
  const [familyLoading, setFamilyLoading] = useState(false);

  useEffect(() => {
    fetchRatingsAndStats();
    if (activeTab === "train") {
      fetchAvailableModels();
    }
    if (activeTab === "models") {
      fetchModelFamilies();
    }
  }, []);

  useEffect(() => {
    if (activeTab === "train") {
      fetchAvailableModels();
    }
    if (activeTab === "models") {
      fetchModelFamilies();
    }
  }, [activeTab]);

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

  const fetchAvailableModels = async () => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
      const res = await fetch(`${backendUrl}/models?token=${authToken}`);
      
      if (!res.ok) {
        throw new Error("Failed to fetch models");
      }
      
      const data = await res.json();
      setModels(data.models || []);
      
      if (data.models && data.models.length > 0) {
        setSelectedModel(data.models[0].name);
      }
    } catch (err) {
      console.error("Error fetching models:", err);
      setTrainingError("Failed to load available models");
    }
  };

  const handleTrainModel = async (e) => {
    e.preventDefault();
    
    if (!selectedModel) {
      setTrainingError("Please select a model");
      return;
    }
    
    if (!trainingFile) {
      setTrainingError("Please select a training file");
      return;
    }
    
    setTrainingLoading(true);
    setTrainingError("");
    setTrainingMessage("");
    
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
      const formData = new FormData();
      formData.append("model_name", selectedModel);
      formData.append("training_file", trainingFile);
      formData.append("token", authToken);
      
      const res = await fetch(`${backendUrl}/train_model?token=${authToken}`, {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to train model");
      }
      
      const data = await res.json();
      setTrainingMessage(`✓ Model training initiated! New model: ${data.new_model}`);
      setTrainingFile(null);
      document.getElementById("training-file-input").value = "";
      // Refresh model families after training
      setTimeout(() => fetchModelFamilies(), 1000);
    } catch (err) {
      setTrainingError(`Error: ${err.message}`);
      console.error(err);
    } finally {
      setTrainingLoading(false);
    }
  };

  const fetchModelFamilies = async () => {
    setFamilyLoading(true);
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
      const res = await fetch(`${backendUrl}/admin/model_families?token=${authToken}`);
      
      if (!res.ok) {
        throw new Error("Failed to fetch model families");
      }
      
      const data = await res.json();
      setModelFamilies(data.model_families || []);
    } catch (err) {
      console.error("Error fetching model families:", err);
    } finally {
      setFamilyLoading(false);
    }
  };

  const handleSetActiveModel = async (baseFamily, modelName) => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
      const formData = new FormData();
      formData.append("base_family", baseFamily);
      formData.append("model_name", modelName);
      
      const res = await fetch(`${backendUrl}/admin/set_active_model?token=${authToken}`, {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) {
        throw new Error("Failed to set active model");
      }
      
      // Refresh the list
      fetchModelFamilies();
      alert(`Active model for ${baseFamily} updated to ${modelName}`);
    } catch (err) {
      alert(`Error: ${err.message}`);
      console.error(err);
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
        
        <h2 className="admin-title">Admin Dashboard</h2>
        
        <div className="admin-tabs">
          <button 
            className={`tab-btn ${activeTab === "ratings" ? "active" : ""}`}
            onClick={() => setActiveTab("ratings")}
          >
            Response Ratings
          </button>
          <button 
            className={`tab-btn ${activeTab === "models" ? "active" : ""}`}
            onClick={() => setActiveTab("models")}
          >
            Model Versions
          </button>
          <button 
            className={`tab-btn ${activeTab === "train" ? "active" : ""}`}
            onClick={() => setActiveTab("train")}
          >
            Train Models
          </button>
        </div>

        {activeTab === "ratings" && (
          <>
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
          </>
        )}

        {activeTab === "models" && (
          <div className="models-management-section">
            <h3>Model Versions Management</h3>
            <p className="section-info">Select which version of each model family users will use</p>
            
            {familyLoading ? (
              <div style={{ textAlign: "center", padding: "40px" }}>
                <p>Loading model families...</p>
              </div>
            ) : modelFamilies.length === 0 ? (
              <div style={{ textAlign: "center", padding: "40px" }}>
                <p>No model families found</p>
              </div>
            ) : (
              <div className="model-families-container">
                {modelFamilies.map((family) => (
                  <div key={family.base_family} className="model-family-card">
                    <h4 className="family-title">
                      {family.base_family.charAt(0).toUpperCase() + family.base_family.slice(1)}
                    </h4>
                    <div className="active-model-display">
                      <strong>Active Version:</strong>
                      <div className="active-model-name">{family.active_model}</div>
                    </div>
                    
                    <div className="versions-list">
                      <p className="versions-label">Available Versions:</p>
                      {family.versions.map((version) => (
                        <div 
                          key={version.model_name} 
                          className={`version-item ${version.is_active ? "active" : ""}`}
                        >
                          <div className="version-info">
                            <div className="version-name">{version.model_name}</div>
                            <div className="version-date">
                              Created: {new Date(version.created_at).toLocaleDateString()}
                            </div>
                            {version.parent_model && (
                              <div className="version-parent">
                                Fine-tuned from: {version.parent_model}
                              </div>
                            )}
                          </div>
                          {!version.is_active && (
                            <button 
                              className="activate-btn"
                              onClick={() => handleSetActiveModel(family.base_family, version.model_name)}
                            >
                              Activate
                            </button>
                          )}
                          {version.is_active && (
                            <div className="active-badge">✓ Active</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === "train" && (
          <div className="training-section">
            <h3>Train a Model</h3>
            
            {trainingError && <div className="admin-error">{trainingError}</div>}
            {trainingMessage && <div className="admin-success">{trainingMessage}</div>}
            
            <form onSubmit={handleTrainModel} className="training-form">
              <div className="form-group">
                <label htmlFor="model-select">Select Model:</label>
                <select
                  id="model-select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="form-control"
                  disabled={models.length === 0}
                >
                  {models.length === 0 ? (
                    <option>Loading models...</option>
                  ) : (
                    models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="training-file-input">Training File (JSONL):</label>
                <input
                  id="training-file-input"
                  type="file"
                  accept=".jsonl,.json"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    setTrainingFile(file || null);
                  }}
                  className="form-control"
                  disabled={trainingLoading}
                />
                <small>Upload a JSONL file with training examples (one JSON object per line)</small>
              </div>

              <div className="form-info">
                <p>💡 <strong>How to prepare JSONL training data:</strong></p>
                <ul>
                  <li>Create a file with one JSON object per line</li>
                  <li>Example format: <code>{`{"role": "user", "content": "question"}`}</code></li>
                  <li>Followed by: <code>{`{"role": "assistant", "content": "answer"}`}</code></li>
                  <li>Or conversation format: <code>{`{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}`}</code></li>
                  <li>All lines must be valid JSON</li>
                </ul>
              </div>

              <button 
                type="submit" 
                className="train-btn"
                disabled={trainingLoading || !selectedModel || !trainingFile}
              >
                {trainingLoading ? "Training..." : "Start Training"}
              </button>
            </form>

            <div className="models-list">
              <h4>Available Models:</h4>
              {models.length === 0 ? (
                <p className="no-models">No models available. Please ensure Ollama is running.</p>
              ) : (
                <ul>
                  {models.map((model) => (
                    <li key={model.name}>
                      <strong>{model.name}</strong>
                      <br />
                      <small>Size: {(model.size / (1024**3)).toFixed(2)} GB</small>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
