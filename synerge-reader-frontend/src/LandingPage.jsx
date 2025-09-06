import TitleLogo from "./components/TitleLogo";
import Top from "./components/Top";
import LandingTop from "./components/LandingTop";
import "./Landing.css"


function LandingPage(){


return(
<div className="LandingView">
<div className="topView">
    <LandingTop></LandingTop>
    <hr/>
 </div>

<div className="main-view">
<div className="box-1">
<TitleLogo></TitleLogo>
<h1>Interactive Human AI Collaborative Reading System</h1>
<button> GET STARTED</button>
</div>

<div className="box-2">
<TitleLogo></TitleLogo>
<h1>Invite</h1>
<button> GET STARTED</button>
</div>
</div>




   <footer>  
  <hr/>     
 <div className="footContents">
   <div><p>© {new Date().getFullYear()} Synergy Reader. All rights reserved. </p> </div>
   </div>
</footer>
</div>
)








}

export default LandingPage