


function SurveyModal({ setOpenSurvey , setNotification }) {
  return (
    <div className="overlay">
      <div className="modal">
        <button className="close-btn" onClick={() => setOpenSurvey(false)}>
          X
        </button>

        <h2 className="modal-title">
          Survey
          <hr />
        </h2>

        <h1 className="guides">How familiar are you with AI services? (1–5)</h1>
   <input style={{height : '2em', fontSize : '1em', color: 'black',padding : "3px"}}
  type="number" 
  min="1" 
  max="5" 
  placeholder="1–5" 
  className="ai-familiarity-input"
/>
         <h1 className="guides">How do you plan to use our service?</h1>
          <input style={{height : '2em', fontSize : '1em', color: 'black',padding : "3px"}}
  placeholder="e.g., Research, Hobby, Education" 
  className="ai-familiarity-input"
/>
         <h1 className="guides">Do you face obstacles using AI?</h1>
          <input style={{height : '2em', fontSize : '1em', color: 'black', padding : "3px"}}

  placeholder="e.g., Language barriers, internet access, cost..." 
  className="ai-familiarity-input"
/>

        <div className="button-group">
          <button className="submit-btn" onClick={() => { 
                setNotification("Thanks for the feedback ◡̈ ")
                setOpenSurvey(false)
            }}>Submit</button>
          <button
            className="submit-btn"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

export default SurveyModal