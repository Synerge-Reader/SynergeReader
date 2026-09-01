import { useState } from "react";
import "./UserAuth.css";
import { GoogleLogin } from "@react-oauth/google";

// Google Sign-In is only shown when a real OAuth client ID is configured —
// see the note where this is used below for why.
const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || "";
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

function LogoMark() {
  return (
    <div className="auth-logo">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3v18M5 21h14" />
        <path d="M5 7l-3 6a3 3 0 006 0l-3-6zM19 7l-3 6a3 3 0 006 0l-3-6z" />
        <path d="M5 7h14M12 3L8 7h8l-4-4z" />
      </svg>
    </div>
  );
}

export default function UserAuth({ setOpenAuth, setAuthToken, setNotification, setOpenSurvey, getHistory }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("")
  // auth: login/register form · forgot: enter email to reset · sent: generic
  // "check your email" confirmation (register/forgot/resend all land here,
  // customized by sentContext) · unverified: login blocked, offer a resend
  const [view, setView] = useState("auth")
  const [sentContext, setSentContext] = useState("register") // register | forgot | resend
  const [authMode, setAuthMode] = useState("login") // login | register
  const [busy, setBusy] = useState(false);

  const handleAuth = async (endpoint) => {
    setBusy(true);
    try {
      let res = ""
      if (endpoint == "register") {
        res = await fetch(`${BACKEND_URL}/${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password, email }),
        });
      }
      else if (endpoint == "login") {
        res = await fetch(`${BACKEND_URL}/${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
      }

      if (res.status == 200) {
        const data = await res.json();
        if (endpoint === "register") {
          // Accounts start unverified — don't log them in yet, send them to
          // the "check your email" screen instead.
          setSentContext("register");
          setView("sent");
        } else {
          setAuthToken(data.token);
          localStorage.setItem("authToken", data.token);
          setOpenAuth(false);
          getHistory();
          setNotification(`Successful ${endpoint}!`);
          setOpenSurvey(true)
        }
      } else {
        let message = endpoint === "login" ? "Could not log in" : "Could not register";
        try {
          const data = await res.json();
          if (data?.detail) message = data.detail;
        } catch { /* no JSON body — keep the generic message */ }

        // A login blocked specifically for an unverified email gets its own
        // screen with a direct "resend" action, instead of just a toast the
        // user has to interpret on their own.
        if (endpoint === "login" && res.status === 403 && /verify your email/i.test(message)) {
          setView("unverified");
        } else {
          setNotification(message);
        }
      }
    } catch (err) {
      console.error("Auth error:", err);
      setNotification("Could not reach the server — check your connection and try again.")
    } finally {
      setBusy(false);
    }
  };

  const handleResendVerification = async () => {
    setBusy(true);
    try {
      await fetch(`${BACKEND_URL}/resend-verification`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      setSentContext("resend");
      setView("sent");
    } catch (err) {
      console.error("Resend error:", err);
      setNotification("Could not reach the server — check your connection and try again.")
    } finally {
      setBusy(false);
    }
  };

  // Handle Google login response
  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      // credentialResponse.credential is the JWT token from Google
      const googleToken = credentialResponse.credential;
      const res = await fetch(`${BACKEND_URL}/google-login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: googleToken }),
      });

      if (res.status === 200) {
        const data = await res.json();
        setAuthToken(data.token);
        localStorage.setItem("authToken", data.token);
        setOpenAuth(false);
        getHistory();
        setNotification("Google login successful!");
        setOpenSurvey(true);
      } else {
        const errorData = await res.json();
        setNotification(`Google login error: ${errorData.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Google login error:", err);
      setNotification("Google login error");
    }
  };

  const handleGoogleError = () => {
    setNotification("Google login failed");
  };

  const handleForgotPassword = async () => {
    setBusy(true);
    try {
      const res = await fetch(`${BACKEND_URL}/forgot-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (res.status == 200) {
        setSentContext("forgot");
        setView("sent");
      } else {
        setNotification("Could not send the reset email — try again.");
      }
    }
    catch (err) {
      console.error("Auth error:", err);
      setNotification("Could not reset password")
    } finally {
      setBusy(false);
    }
  }

  const canSubmit = authMode === "register"
    ? username.trim() && email.trim() && password.trim()
    : username.trim() && password.trim();

  const SENT_COPY = {
    register: { title: "Check your email", body: `We sent a verification link to ${email || "your email"}. Click it to activate your account, then come back and log in.` },
    forgot: { title: "Check your email", body: `If ${email || "that email"} is registered, we sent a link to reset your password. It expires in 1 hour.` },
    resend: { title: "Check your email", body: `If ${email || "that email"} exists and isn't verified yet, we sent a new verification link.` },
  };

  return (
    <div className="overlay" onClick={() => setOpenAuth(false)}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="close-btn" onClick={() => setOpenAuth(false)} aria-label="Close">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
        </button>

        <div className="auth-header">
          <LogoMark />
          <div className="auth-brand">SynergeReader</div>
          {view === "auth" && (
            <>
              <h2 className="auth-title">{authMode === "login" ? "Welcome back" : "Create your account"}</h2>
              <p className="auth-subtitle">
                {authMode === "login" ? "Sign in to continue to your workspace." : "Get started with your firm's legal AI workspace."}
              </p>
            </>
          )}
          {view === "forgot" && (
            <>
              <h2 className="auth-title">Reset your password</h2>
              <p className="auth-subtitle">We'll email you a link to choose a new one.</p>
            </>
          )}
          {view === "sent" && (
            <>
              <h2 className="auth-title">{SENT_COPY[sentContext].title}</h2>
              <p className="auth-subtitle">{SENT_COPY[sentContext].body}</p>
            </>
          )}
          {view === "unverified" && (
            <>
              <h2 className="auth-title">Verify your email first</h2>
              <p className="auth-subtitle">This account hasn't confirmed its email yet. Enter it below and we'll resend the verification link.</p>
            </>
          )}
        </div>

        {view === "auth" && (
          <div className="auth-tabs">
            <button
              className={authMode === "login" ? "auth-tab active" : "auth-tab"}
              onClick={() => setAuthMode("login")}
            >Log In</button>
            <button
              className={authMode === "register" ? "auth-tab active" : "auth-tab"}
              onClick={() => setAuthMode("register")}
            >Register</button>
          </div>
        )}

        {view === "auth" && (
          <>
            <div className="auth-field">
              {/* Login accepts either the username or the account's email
                  (backend now matches both) — Register still asks for a
                  fresh username specifically, since email has its own field
                  right below. */}
              <label className="auth-label">{authMode === "login" ? "Username or email" : "Username"}</label>
              <input
                className="auth-input"
                placeholder={authMode === "login" ? "john or you@firm.com" : "John Doe"}
                type="text"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>

            {/* Email ONLY for register */}
            {authMode === "register" && (
              <div className="auth-field">
                <label className="auth-label">Email</label>
                <input
                  className="auth-input"
                  placeholder="you@firm.com"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            )}

            <div className="auth-field">
              <div className="auth-label-row">
                <label className="auth-label">Password</label>
                {authMode === "login" && (
                  <button type="button" className="auth-link-inline" onClick={() => setView("forgot")}>Forgot?</button>
                )}
              </div>
              <input
                className="auth-input"
                placeholder="••••••••"
                type="password"
                autoComplete={authMode === "login" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && canSubmit && !busy) handleAuth(authMode); }}
              />
              {authMode === "register" && (
                <div style={{ fontSize: "11px", color: "#9ca3af", marginTop: "5px" }}>At least 8 characters.</div>
              )}
            </div>

            <button
              className="auth-submit"
              disabled={!canSubmit || busy}
              onClick={() => handleAuth(authMode)}
            >
              {busy ? "Please wait…" : authMode === "login" ? "Log In" : "Create Account"}
            </button>

            <div className="auth-switch">
              {authMode === "login" ? "Don't have an account?" : "Already have an account?"}{" "}
              <button
                className="auth-link-inline"
                onClick={() => setAuthMode(authMode === "login" ? "register" : "login")}
              >{authMode === "login" ? "Register" : "Log in"}</button>
            </div>

            {/* Google Sign-In only renders when a real client ID is configured —
                with no client ID, Google's own page rejects the request with an
                error, so showing the button at all would just be broken. Once
                REACT_APP_GOOGLE_CLIENT_ID is set (see backend .env for the
                matching GOOGLE_CLIENT_ID), this reappears automatically. */}
            {GOOGLE_CLIENT_ID && (
              <>
                <div className="auth-divider"><span>Or continue with</span></div>
                <div className="auth-google">
                  <GoogleLogin
                    onSuccess={handleGoogleSuccess}
                    onError={handleGoogleError}
                    text="signin_with"
                    size="large"
                  />
                </div>
              </>
            )}
          </>
        )}

        {view === "forgot" && (
          <>
            <div className="auth-field">
              <label className="auth-label">Email</label>
              <input
                className="auth-input"
                placeholder="you@firm.com"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && email.trim() && !busy) handleForgotPassword(); }}
              />
            </div>

            <button className="auth-submit" disabled={!email.trim() || busy} onClick={handleForgotPassword}>
              {busy ? "Sending…" : "Send Reset Link"}
            </button>
            <button className="auth-secondary" onClick={() => setView("auth")}>Back to Log In</button>
          </>
        )}

        {view === "unverified" && (
          <>
            <div className="auth-field">
              <label className="auth-label">Email</label>
              <input
                className="auth-input"
                placeholder="you@firm.com"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && email.trim() && !busy) handleResendVerification(); }}
              />
            </div>
            <button className="auth-submit" disabled={!email.trim() || busy} onClick={handleResendVerification}>
              {busy ? "Sending…" : "Resend Verification Email"}
            </button>
            <button className="auth-secondary" onClick={() => setView("auth")}>Back to Log In</button>
          </>
        )}

        {view === "sent" && (
          <button className="auth-submit" onClick={() => { setView("auth"); setAuthMode("login"); }}>Back to Log In</button>
        )}
      </div>
    </div>
  );
}
