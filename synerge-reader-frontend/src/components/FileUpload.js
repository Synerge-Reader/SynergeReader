import React, { useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import { GlobalWorkerOptions } from "pdfjs-dist/build/pdf";
import mammoth from "mammoth";
import Dropdown from "./Dropdown";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const FileUpload = ({ onFileParsed, setIsLoading, setError, model, setModel }) => {
  const fileInputRef = useRef();
  const [isDragging, setIsDragging] = useState(false);
  const allowedTypes = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
  const setDefault = () => setIsDragging(false);

  const uploadToBackend = async (textContent, fileName) => {
    try {
      const formData = new FormData();
      const blob = new Blob([textContent], { type: 'text/plain' });
      formData.append('file', blob, fileName);

      const response = await fetch(process.env.REACT_APP_BACKEND_URL + '/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Document uploaded successfully:', result);
      return result;
    } catch (error) {
      console.error('Upload error:', error);
      setError(`Failed to upload document: ${error.message}`);
      throw error;
    }
  };

  const processFile = async (file) => {
    if (file.size > 20 * 1024 * 1024) {
      setError("File too large (max 20MB).");
      return;
    }
    if (!allowedTypes.includes(file.type)) {
      setError("Unsupported file type. Please upload PDF, DOCX, or TXT only.");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      let textContent = "";

      if (file.type === "application/pdf") {
        textContent = await processPDF(file);
      } else if (file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
        textContent = await processDOCX(file);
      } else if (file.type === "text/plain") {
        textContent = await processTXT(file);
      }

      // Upload to backend for processing
      await uploadToBackend(textContent, file.name);

      // Call the callback with parsed text
      onFileParsed(textContent, file.name);

    } catch (error) {
      setError(`Error processing file: ${error.message}`);
      setIsLoading(false);
    }
  };

  const processPDF = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        const loadingTask = pdfjsLib.getDocument({ data: e.target.result });
        loadingTask.promise
          .then(async (pdf) => {
            let text = "";
            for (let i = 1; i <= pdf.numPages; i++) {
              const page = await pdf.getPage(i);
              const content = await page.getTextContent();
              text += content.items.map((item) => item.str).join(" ") + "\n";
            }
            resolve(text);
          })
          .catch((error) => {
            reject(new Error(`Error reading PDF: ${error.message}`));
          });
      };
      reader.readAsArrayBuffer(file);
    });
  };

  const processDOCX = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        mammoth.extractRawText({ arrayBuffer: e.target.result })
          .then((result) => {
            resolve(result.value);
          })
          .catch((error) => {
            reject(new Error(`Error reading DOCX: ${error.message}`));
          });
      };
      reader.readAsArrayBuffer(file);
    });
  };

  const processTXT = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        resolve(e.target.result);
      };
      reader.onerror = () => {
        reject(new Error("Error reading text file"));
      };
      reader.readAsText(file, "utf-8");
    });
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      processFile(e.target.files[0]);
      setDefault();
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDefault();
  };

  return (
    <div
      className={`alpha-upload-card${isDragging ? " dragging" : ""}`}
      onDragOver={handleDragEnter}
      onDrop={handleDrop}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      role="region"
      aria-label="File upload area"
    >
      <svg
        className="alpha-upload-icon"
        width="40"
        height="40"
        viewBox="0 0 24 24"
        fill="none"
        stroke="#5563d7"
        strokeWidth="2"
      >
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
      </svg>
      <div className="alpha-upload-hint">
        <strong>Drag & Drop</strong> a <span className="pdf-accent">PDF</span>,{" "}
        <span className="docx-accent">DOCX</span>, or{" "}
        <span className="txt-accent">TXT</span> file here{" "}
        <span className="dim">(max 20MB)</span>
        <br /> or
      </div>
      <input
        type="file"
        accept=".pdf,.docx,.txt"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
        id="file-upload"
      />
      <button
        className="alpha-upload-btn"
        onClick={() => fileInputRef.current.click()}
        aria-label="Browse files for upload"
      >
        Browse Files
      </button>

      <Dropdown
        title={`Selected Model: ${model}`}
        options={["gpt-oss:20b"]}
        onSelect={(option) => {
          setModel(option);
          console.log("Selected:", option);
        }}
      />

    </div>
  );
};

export default FileUpload;