import { useState } from "react";
import "./UserAuth.css";

export default function UserAuth({ setOpenAuth }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async () => {
    const res = await fetch(
      (process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") +
      "/register",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username,
          password: password
        }),
      },
    );
    console.log(res);
  };

  return (
    <div className="overlay">
      <div className="modal">
        {/* Close button */}
        <button className="close-btn" onClick={() => setOpenAuth(false)}>
          ×
        </button>
        <h2 className="modal-title">Log In</h2>
        {/*Username and password*/}
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
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {/* Submit Button */}
        <button className="submit-btn" onClick={() => handleSubmit()}>
          Submit
        </button>
      </div>
    </div>
  );
}
