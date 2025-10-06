import React, { useState, useRef, useEffect } from "react";
import "./Dropdown.css";

const Dropdown = ({ title, options, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="dropdown-container">
      <div className="dropdown" ref={dropdownRef}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="dropdown-btn"
        >
          {title} <span className="arrow">▾</span>
        </button>
        {isOpen && (
          <div className="dropdown-menu">
            {options.map((opt, idx) => (
              <button
                key={idx}
                onClick={() => {
                  onSelect(opt);
                  setIsOpen(false);
                }}
                className="dropdown-item"
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dropdown;
