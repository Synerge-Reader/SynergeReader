import React, { useEffect } from "react";
import "./Notifier.css";

export default function Notifier({ message, duration = 3000, setNotification }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      setNotification('');
    }, duration);

    return () => clearTimeout(timer);
  }, [duration]);

  if (!message) return null;

  return <div className="notifier">{message}</div>;
}
