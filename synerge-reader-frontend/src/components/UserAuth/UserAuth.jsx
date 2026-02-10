import { useState } from "react";
import "./UserAuth.css";
import { GoogleLogin } from "@react-oauth/google";

export default function UserAuth({ setOpenAuth, setAuthToken, setNotification, setOpenSurvey, getHistory }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("")
  const [view, setView] = useState("auth") // auth | forgot
  const [authMode, setAuthMode] = useState("login") // login | register

  const handleAuth = async (endpoint) => {
    try {
      let res = ""
      if (endpoint == "register") {
        res = await fetch(
          (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") +
          `/${endpoint}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              username,
              password,
              email
            }),
          }

        );
      }
      else if (endpoint == "login") {

        res = await fetch(
          (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") +
          `/${endpoint}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              username,
              password,
            }),
          }

        );


      }

      if (res.status == 200) {
        const data = await res.json();
        setAuthToken(data.token);
        localStorage.setItem("authToken", data.token);
        setOpenAuth(false);
        getHistory();
        setNotification(`Successful ${endpoint}!`);
        setOpenSurvey(true)
      } else {
        setNotification("User Auth Error")
      }
    } catch (err) {
      console.error("Auth error:", err);
      setNotification("User Auth Error")
    }
  };

  // Handle Google login response
  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      // credentialResponse.credential is the JWT token from Google
      const googleToken = credentialResponse.credential;

      // Send token to backend for verification
      const res = await fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + "/google-login",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token: googleToken }),
        }
      );

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
    try {
      let endpoint = "forgot-password"
      const res = await fetch(
        (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") +
        `/${endpoint}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email
          }),
        }

      );
      if (res.status == 200) {
        setNotification(`Successful! New password sent to your email`);
        setView("auth")
      }
    }
    catch (err) {
      console.error("Auth error:", err);
      setNotification("Could not reset password")

    }
  }




  return (


    <div className="overlay">
      <div className="modal">
        {/* Close button */}
        <button className="close-btn" onClick={() => setOpenAuth(false)}>
          X
        </button>
        <h2 className="modal-title">User Portal
          <hr></hr></h2>




        {/* Username and password OR Forgot Password */}
        {view === "auth" ? (
          <>
            {/* Username */}
            <h1 className="guides">Username</h1>
            <input
              className="comment-box"
              rows="1"
              placeholder="John Doe"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />

            {/* Email ONLY for register */}
            {authMode === "register" && (
              <>
                <h1 className="guides">Email</h1>
                <input
                  className="comment-box"
                  rows="1"
                  placeholder="Enter your email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </>
            )}

            {/* Password */}
            <h1 className="guides">Password</h1>
            <input

              className="comment-box"
              rows="1"
              placeholder="*******"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            {/* Buttons */}
            <div className="button-group">
              {authMode === "login" ? (
                <button
                  className="submit-btn"
                  onClick={() => handleAuth("login")}
                >
                  Log In
                </button>
              ) : (
                <button
                  className="submit-btn"
                  onClick={() => handleAuth("register")}
                >
                  Register
                </button>
              )}
            </div>

            {/* Switch login/register */}
            <button
              onClick={() =>
                setAuthMode(authMode === "login" ? "register" : "login")
              }
              style={{
                background: "none",
                border: "none",
                color: "#007bff",
                cursor: "pointer",
                marginTop: "10px"
              }}
            >
              {authMode === "login"
                ? "Don't have an account? Register"
                : "Already have an account? Log in"}
            </button>

            {/* Forgot password */}
            {authMode === "login" && (
              <button
                onClick={() => setView("forgot")}
                style={{
                  background: "none",
                  border: "none",
                  color: "#007bff",
                  cursor: "pointer",
                  marginTop: "10px"
                }}
              >
                Forgot password?
              </button>
            )}
          </>
        ) : (
          <>
            {/* Forgot Password View */}
            <h1 className="guides">Email</h1>
            <input
              className="comment-box"
              rows="1"
              placeholder="Enter your email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />

            <div className="button-group">
              <button
                className="submit-btn"
                onClick={handleForgotPassword}
              >
                Reset Password
              </button>

              <button
                className="submit-btn"
                onClick={() => setView("auth")}
              >
                Back to Login
              </button>
            </div>
          </>
        )}

        {/* Google Login Section */}
        <div style={{ marginTop: "20px", textAlign: "center" }}>
          <hr style={{ margin: "15px 0" }} />
          <p style={{ color: "#666", marginBottom: "15px", fontSize: "14px" }}>
            Or sign in with Google
          </p>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={handleGoogleError}
              text="signin_with"
              size="large"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
