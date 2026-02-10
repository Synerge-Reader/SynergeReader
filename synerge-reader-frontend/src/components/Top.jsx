import "./Top.css";
import { useState, useEffect } from "react";

// Model to Domain mapping
const MODEL_DOMAIN_MAP = {
  "llama3.1:8b": "General Domain",
  "OussamaELALLAM/MedExpert:latest": "Medical Domain",
  "adrienbrault/saul-instruct-v1:Q8_0": "Legal Domain",
};

const MODEL_OPTIONS = [
  { label: "LLaMA 3.1 8B", value: "llama3.1:8b", domain: "General Domain" },
  { label: "MedExpert", value: "OussamaELALLAM/MedExpert:latest", domain: "Medical Domain" },
  { label: "Saul Instruct", value: "adrienbrault/saul-instruct-v1:Q8_0", domain: "Legal Domain" },
];

function Top({ setOpenAuth, authToken, setAuthToken, setHistory, model, setModel, onAdminClick, onModelChange }) {
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);

  // Derive domain from current model
  const currentDomain = MODEL_DOMAIN_MAP[model] || "General Domain";
  const currentModelLabel = MODEL_OPTIONS.find(opt => opt.value === model)?.label || model;

  useEffect(() => {
    if (authToken) {
      checkAdminStatus();
    }
  }, [authToken]);

  const checkAdminStatus = async () => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
      const res = await fetch(`${backendUrl}/admin/check?token=${authToken}`);
      const data = await res.json();
      setIsAdmin(data.is_admin);
    } catch (err) {
      console.error("Error checking admin status:", err);
      setIsAdmin(false);
    }
  };

  const handleModelSelect = (modelValue) => {
    const previousModel = model;
    setModel(modelValue);
    setShowModelDropdown(false);

    // Notify parent about model change (for mid-conversation tracking)
    if (onModelChange && previousModel !== modelValue) {
      onModelChange(previousModel, modelValue);
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (showModelDropdown && !e.target.closest('.model-selector')) {
        setShowModelDropdown(false);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [showModelDropdown]);

  return (
    <>
      <div className="head">
        <div className="headingInfo">
          <img src="/menuBar.svg" alt="menu" />

          {/* Domain Display (Auto-assigned based on model) */}
          <div className="domain-display" style={{
            marginLeft: "auto",
            padding: "6px 12px",
            background: currentDomain === "Medical Domain" ? "#e8f5e9" :
              currentDomain === "Legal Domain" ? "#fff3e0" : "#e3f2fd",
            borderRadius: "6px",
            fontSize: "0.85rem",
            fontWeight: "500",
            color: currentDomain === "Medical Domain" ? "#2e7d32" :
              currentDomain === "Legal Domain" ? "#e65100" : "#1565c0",
          }}>
            {currentDomain === "Medical Domain" && "🏥 "}
            {currentDomain === "Legal Domain" && "⚖️ "}
            {currentDomain === "General Domain" && "🌐 "}
            {currentDomain}
          </div>

          {/* Model Selector Dropdown */}
          <div className="model-selector" style={{ position: "relative", marginLeft: "12px" }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowModelDropdown((prev) => !prev);
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 14px",
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontSize: "0.9rem",
                fontWeight: "500",
                boxShadow: "0 2px 8px rgba(102, 126, 234, 0.3)",
                transition: "all 0.2s ease"
              }}
              title="Change AI Model"
            >
              <span>🤖</span>
              <span>{currentModelLabel}</span>
              <span style={{ fontSize: "0.7rem" }}>▼</span>
            </button>

            {showModelDropdown && (
              <div
                className="dropdown-menu"
                style={{
                  position: "absolute",
                  top: "calc(100% + 4px)",
                  right: 0,
                  background: "#fff",
                  border: "1px solid #e0e0e0",
                  borderRadius: "8px",
                  padding: "8px 0",
                  zIndex: 1000,
                  minWidth: "220px",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
                }}
              >
                <div style={{
                  padding: "6px 14px",
                  fontSize: "0.75rem",
                  color: "#888",
                  borderBottom: "1px solid #eee",
                  marginBottom: "4px"
                }}>
                  Select AI Model
                </div>
                {MODEL_OPTIONS.map((option) => (
                  <div
                    key={option.value}
                    onClick={() => handleModelSelect(option.value)}
                    style={{
                      padding: "10px 14px",
                      cursor: "pointer",
                      whiteSpace: "nowrap",
                      background: model === option.value ? "#f5f5ff" : "transparent",
                      borderLeft: model === option.value ? "3px solid #667eea" : "3px solid transparent",
                      transition: "all 0.15s ease",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                    onMouseEnter={(e) => {
                      if (model !== option.value) {
                        e.currentTarget.style.background = "#f8f8f8";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (model !== option.value) {
                        e.currentTarget.style.background = "transparent";
                      }
                    }}
                  >
                    <span style={{ fontWeight: model === option.value ? "600" : "400" }}>
                      {option.label}
                    </span>
                    <span style={{
                      fontSize: "0.75rem",
                      color: "#888",
                      background: "#f0f0f0",
                      padding: "2px 6px",
                      borderRadius: "4px"
                    }}>
                      {option.domain.replace(" Domain", "")}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <h3> <a
            href="/SynergeReader User Guide.pdf"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: "none", color: "inherit" }}
            g >  User guide</a>
            </h3>
          <h3>Community Papers</h3>

          {isAdmin && authToken && (
            <button
              onClick={onAdminClick}
              className="admin-btn"
              title="Admin Dashboard"
            >
              📊 Admin
            </button>
          )}

          {!authToken ? (
            <div onClick={() => setOpenAuth(true)} className="auth">
              <img className="accLogo" src="/accountIcon.svg" alt="account" />
            </div>
          ) : (
            <div
              className="auth"
              onClick={() => {
                setAuthToken("");
                localStorage.setItem("authToken", "");
                setHistory([]);
              }}
            >
              Sign out
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default Top;
