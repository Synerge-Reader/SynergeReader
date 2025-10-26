import "./Top.css";
import { useState } from "react";


function Top({ setOpenAuth, authToken, setAuthToken, setHistory, model, setModel }) {

   const [selectedDomain, setSelectedDomain] = useState("Domains");
  const [showDropdown, setShowDropdown] = useState(false);

  const handleSelect = (domain) => {
    setSelectedDomain(domain);
    setShowDropdown(false);

    if (domain === "Medical Domain") {
    setModel("OussamaELALLAM/MedExpert:latest"); // this line might cause trouble
  } else if (domain === "Legal Domain") {
    setModel("adrienbrault/saul-instruct-v1:Q8_0");
  } else {
    setModel("llama3.1:8b");
  }

  };

  return (
   <>
      <div className="head">
        <div className="headingInfo">
          <img src="/menuBar.svg" alt="menu" />

          <div className="domain-dropdown" style={{ position: "relative", marginLeft: "auto" }}>
            <h3
              onClick={() => setShowDropdown((prev) => !prev)}
              style={{ cursor: "pointer", userSelect: "none" }}
            >
              {selectedDomain}
            </h3>

            {showDropdown && (
              <div
                className="dropdown-menu"
                style={{
                  position: "absolute",
                  top: "100%",
                  right: 0,
                  background: "#fff",
                  border: "1px solid #ccc",
                  borderRadius: "5px",
                  padding: "5px 0",
                  zIndex: 10,
                }}
              >
                {["Medical Domain", "Legal Domain"].map((domain) => (
                  <div
                    key={domain}
                    onClick={() => handleSelect(domain)}
                    style={{
                      padding: "8px 15px",
                      cursor: "pointer",
                      whiteSpace: "nowrap",
                      background:
                        selectedDomain === domain ? "#f0f0f0" : "transparent",
                    }}
                  >
                    {domain}
                  </div>
                ))}
              </div>
            )}
          </div>

          <h3>Community Papers</h3>

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
