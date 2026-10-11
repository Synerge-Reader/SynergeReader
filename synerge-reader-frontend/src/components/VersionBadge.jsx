import { useEffect, useState } from "react";

const DEFAULT_BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const VERSION_UNAVAILABLE = "Version unavailable";

const SEMVER = /^\d+\.\d+\.\d+$/;
const SHORT_SHA = /^[0-9a-f]{7,40}$/;

// The label comes only from the server's /version response. The frontend's
// own build, package version and Git state are never consulted: what this
// shows is what the running backend says it is. A response that does not
// have the expected shape reads as unavailable rather than being guessed at.
export function versionLabel(payload) {
  if (!payload || typeof payload !== "object") return VERSION_UNAVAILABLE;
  const { app_version: appVersion, build_known: buildKnown, build_sha_short: shortSha } = payload;
  if (typeof appVersion !== "string" || !SEMVER.test(appVersion)) return VERSION_UNAVAILABLE;
  if (buildKnown === true && typeof shortSha === "string" && SHORT_SHA.test(shortSha)) {
    return `v${appVersion} · ${shortSha}`;
  }
  if (buildKnown === false) return `v${appVersion} · build unknown`;
  return VERSION_UNAVAILABLE;
}

const visuallyHidden = {
  position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px",
  overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", border: 0,
};

/**
 * Small application-version line for the app chrome. Fetches once, never
 * blocks rendering of anything else, and renders nothing until the server
 * has answered (or failed to).
 */
export default function VersionBadge({ backendUrl = DEFAULT_BACKEND, style }) {
  const [label, setLabel] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${backendUrl}/version`, { cache: "no-store" })
      .then(res => {
        if (!res.ok) throw new Error("version request failed");
        return res.json();
      })
      .then(payload => { if (!cancelled) setLabel(versionLabel(payload)); })
      .catch(() => { if (!cancelled) setLabel(VERSION_UNAVAILABLE); });
    return () => { cancelled = true; };
  }, [backendUrl]);

  if (label === null) return null;
  return (
    <div data-testid="app-version" style={{ position: "relative", ...style }}>
      <span style={visuallyHidden}>Application version: </span>
      {label}
    </div>
  );
}
