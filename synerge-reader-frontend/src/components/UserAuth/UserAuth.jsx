import { useState } from "react";
import "./UserAuth.css";

export default function UserAuth({ setOpenAuth, setAuthToken, setNotification, setOpenSurvey, getHistory }) {
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

  return (
    <div className="overlay">
      <div className="modal">
        {/* Close button */}
        <button className="close-btn" onClick={() => setOpenAuth(false)}>
          X
        </button>
        <h2 className="modal-title">User Portal
        <hr></hr></h2>
      

        {/* Username and password */}
        <h1 className="guides">Username</h1>
        <input
          className="comment-box"
          rows="1"
          placeholder="John Doe"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
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

