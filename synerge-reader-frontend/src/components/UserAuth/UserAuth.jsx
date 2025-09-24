import { useState } from "react";
import "./UserAuth.css";

export default function UserAuth({ setOpenAuth, setAuthToken }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleAuth = async (endpoint) => {
    try {
      const res = await fetch(
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

      const data = await res.json();
      console.log(endpoint.toUpperCase(), data);
      setAuthToken(data.token);
      localStorage.setItem("authToken", data.token);
    } catch (err) {
      console.error("Auth error:", err);
    }
  };

  return (
    <div className="overlay">
      <div className="modal">
        {/* Close button */}
        <button className="close-btn" onClick={() => setOpenAuth(false)}>
          ×
        </button>
        <h2 className="modal-title">Authentication</h2>

        {/* Username and password */}
        <textarea
          className="comment-box"
          rows="1"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <textarea
          className="comment-box"
          rows="1"
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {/* Buttons */}
        <div className="button-group">
          <button
            className="submit-btn"
            onClick={() => handleAuth("login")}
          >
            Log In
          </button>
          <button
            className="submit-btn"
            onClick={() => handleAuth("register")}
          >
            Register
          </button>
        </div>
      </div>
    </div>
  );
}

