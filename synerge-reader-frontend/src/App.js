import { useState, useEffect, useCallback } from "react";
import LandingPage from "./components/LandingPage/LandingPage";
import GridApp from "./GridApp";
import VerifyEmailPage from "./components/AuthAction/VerifyEmailPage";
import ResetPasswordPage from "./components/AuthAction/ResetPasswordPage";

function routeFromHash(hash) {
  if (hash.startsWith("#verify-email")) return "verify-email";
  if (hash.startsWith("#reset-password")) return "reset-password";
  if (hash === "#app") return "app";
  return "landing";
}

/**
 * Top-level switch: the marketing landing page at "/", the real app once
 * you click through, plus two standalone landing targets for the links
 * emailed by registration/forgot-password. No router dependency — hash
 * prefixes are enough for four static destinations, and it keeps every
 * route bookmarkable and reload-safe.
 */
export default function App() {
  const [route, setRoute] = useState(() => routeFromHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setRoute(routeFromHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const enterApp = useCallback(() => {
    window.location.hash = "app";
    setRoute("app");
  }, []);

  if (route === "verify-email") return <VerifyEmailPage />;
  if (route === "reset-password") return <ResetPasswordPage />;
  if (route === "app") return <GridApp />;
  return <LandingPage onEnter={enterApp} />;
}
