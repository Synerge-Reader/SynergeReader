import "./Top.css";

function Top({ setOpenAuth, authToken, setAuthToken, setHistory }) {
  return (
    <>
      <div className="head">
        <div className="headingInfo">
          <img src="/menuBar.svg"></img>
          <h3>Community Papers</h3>
          <h3>Browse</h3>
          {!authToken ? (
            <div onClick={() => setOpenAuth(true)} className='auth'>
              <img className="accLogo" src="/accountIcon.svg"></img>
            </div>
          ) : (
            <div className='auth' onClick={() => { setAuthToken(''); localStorage.setItem("authToken", ""); setHistory([]); }}>
              Sign out
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default Top;
