import { useState } from "react";
import "../UserAuth/UserAuth.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

function getTokenFromHash() {
  const query = window.location.hash.split("?")[1] || "";
  return new URLSearchParams(query).get("token") || "";
}

/**
 * Landing target for the link emailed by /forgot-password:
 * #reset-password?token=... . A real form (not an auto-action like email
 * verification) since it needs the new password from the user.
 */
export default function ResetPasswordPage() {
  const [token] = useState(getTokenFromHash);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState(token ? "" : "This reset link is missing its token.");

  const canSubmit = password.length >= 8 && password === confirm && !busy;

  const submit = async () => {
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    if (password !== confirm) { setError("Passwords don't match."); return; }
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${BACKEND_URL}/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, new_password: password }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        setDone(true);
      } else {
        setError(data.detail || "Could not reset your password.");
      }
    } catch {
      setError("Could not reach the server — check your connection and try again.");
    } finally {
      setBusy(false);
    }
  };

  const goToSignIn = () => { window.location.hash = ""; window.location.reload(); };

  return (
    <div className="auth-standalone">
      <div className="modal">
        <div className="auth-header">
          <div className="auth-logo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3v18M5 21h14" />
              <path d="M5 7l-3 6a3 3 0 006 0l-3-6zM19 7l-3 6a3 3 0 006 0l-3-6z" />
              <path d="M5 7h14M12 3L8 7h8l-4-4z" />
            </svg>
          </div>
          <div className="auth-brand">SynergeReader</div>
          <h2 className="auth-title">{done ? "Password updated" : "Choose a new password"}</h2>
          <p className="auth-subtitle">
            {done ? "You can now log in with your new password." : "Must be at least 8 characters."}
          </p>
        </div>

        {done ? (
          <button className="auth-submit" onClick={goToSignIn}>Continue to Sign In</button>
        ) : token ? (
          <>
            <div className="auth-field">
              <label className="auth-label">New password</label>
              <input
                className="auth-input"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoFocus
              />
            </div>
            <div className="auth-field">
              <label className="auth-label">Confirm new password</label>
              <input
                className="auth-input"
                type="password"
                placeholder="••••••••"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && canSubmit) submit(); }}
              />
            </div>
            {error && <div style={{ color: "#dc2626", fontSize: "12.5px", marginBottom: "10px" }}>{error}</div>}
            <button className="auth-submit" disabled={!canSubmit} onClick={submit}>
              {busy ? "Updating…" : "Update Password"}
            </button>
          </>
        ) : (
          <>
            {error && <div style={{ color: "#dc2626", fontSize: "12.5px", marginBottom: "10px" }}>{error}</div>}
            <button className="auth-submit" onClick={goToSignIn}>Back to SynergeReader</button>
          </>
        )}
      </div>
    </div>
  );
}
