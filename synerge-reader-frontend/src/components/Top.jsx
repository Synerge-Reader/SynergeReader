function Top({ setOpenAuth }) {
  return (
    <>
      <div className="head">
        <div className="headingInfo">
          <img src="/menuBar.svg"></img>
          <h3>Community Papers</h3>
          <h3>Browse</h3>
          <div onClick={() => setOpenAuth(true)}>
            <img className="accLogo" src="/accountIcon.svg"></img>
          </div>
        </div>
      </div>
    </>
  );
}

export default Top;
