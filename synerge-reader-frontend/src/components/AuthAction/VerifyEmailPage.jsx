import { useEffect, useState } from "react";
import "../UserAuth/UserAuth.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

/**
 * Landing target for the link emailed by /register and /resend-verification:
 * #verify-email?token=... . Reads the token straight out of the hash (no
 * router needed for one page) and calls the backend once on mount.
 */
export default function VerifyEmailPage() {
  const [status, setStatus] = useState("checking"); // checking | ok | error
  const [message, setMessage] = useState("Confirming your email…");

  useEffect(() => {
    const query = window.location.hash.split("?")[1] || "";
    const token = new URLSearchParams(query).get("token");
    if (!token) {
      setStatus("error");
      setMessage("This verification link is missing its token.");
      return;
    }
    fetch(`${BACKEND_URL}/verify-email?${new URLSearchParams({ token })}`)
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          setStatus("ok");
          setMessage(data.message || "Your email is verified.");
        } else {
          setStatus("error");
          setMessage(data.detail || "This verification link is invalid or has expired.");
        }
      })
      .catch(() => {
        setStatus("error");
        setMessage("Could not reach the server — check your connection and try again.");
      });
  }, []);

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
          <h2 className="auth-title">
            {status === "checking" ? "Verifying…" : status === "ok" ? "Email verified" : "Verification failed"}
          </h2>
          <p className="auth-subtitle">{message}</p>
        </div>

        {status !== "checking" && (
          <button className="auth-submit" onClick={goToSignIn}>
            {status === "ok" ? "Continue to Sign In" : "Back to SynergeReader"}
          </button>
        )}
      </div>
    </div>
  );
}
