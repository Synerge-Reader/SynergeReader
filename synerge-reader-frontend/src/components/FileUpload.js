import React, { useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import { GlobalWorkerOptions } from "pdfjs-dist/build/pdf";
import mammoth from "mammoth";
import Dropdown from "./Dropdown/Dropdown.jsx";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

export default function FileUpload({
  onFileParsed,
  setIsLoading,
  setError,
  model,
  setModel,
}) {
  const fileInputRef = useRef();
  const [isDragging, setIsDragging] = useState(false);
  const allowedTypes = [
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/json"
  ];
  const setDefault = () => setIsDragging(false);

  const uploadBatchToBackend = async (parsedDocs) => {
    try {
      const formData = new FormData();
      parsedDocs.forEach(({ text, name }) => {
        const blob = new Blob([text], { type: 'text/plain' });
        formData.append('files', blob, name);
      });

      const response = await fetch((process.env.REACT_APP_BACKEND_URL || "http://localhost:5000") + '/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Documents uploaded successfully:', result);
      return result;
    } catch (error) {
      console.error('Upload error:', error);
      setError(`Failed to upload documents: ${error.message}`);
      throw error;
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
        mammoth
          .extractRawText({ arrayBuffer: e.target.result })
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

  const processJSON = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target.result);
          resolve(JSON.stringify(json, null, 2));
        } catch (error) {
          reject(new Error(`Error reading JSON: ${error.message}`));
        }
      };
      reader.readAsText(file, "utf-8");
    });
  };

  const processFiles = async (files) => {
    const validFiles = [];
    const errorMessages = [];

    files.forEach((file) => {
      if (file.size > 20 * 1024 * 1024) {
        errorMessages.push(`${file.name} is too large (max 20MB).`);
        return;
      }
      if (!allowedTypes.includes(file.type)) {
        errorMessages.push(`${file.name} has an unsupported type. Please upload PDF, DOCX, or TXT only.`);
        return;
      }
      validFiles.push(file);
    });

    if (errorMessages.length) {
      setError(errorMessages.join(" "));
    } else {
      setError("");
    }

    if (validFiles.length === 0) {
      return;
    }

    setIsLoading(true);

    const parsedDocs = [];

    try {
      for (const file of validFiles) {
        let textContent = "";

        if (file.type === "application/pdf") {
          textContent = await processPDF(file);
        } else if (file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
          textContent = await processDOCX(file);
        } else if (file.type === "text/plain") {
          textContent = await processTXT(file);
        }
        if (file.type === "application/json") {
          textContent = await processJSON(file);
        }

        const fileUrl = URL.createObjectURL(file);

        const parsedDoc = {
          name: file.name,
          text: textContent,
          url: fileUrl,
          type: file.type
        };
        parsedDocs.push(parsedDoc);
        onFileParsed(parsedDoc);
      }

      const backendResults = await uploadBatchToBackend(parsedDocs);

      const uploadErrors = [];
      backendResults.forEach((result, index) => {
        if (result && result.error) {
          uploadErrors.push(`${parsedDocs[index].name}: ${result.error}`);
        }
      });

      if (uploadErrors.length) {
        setError(uploadErrors.join(" "));
      }
    } catch (error) {
      setError(`Error processing file(s): ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (e) => {
    if (e.target.files.length > 0) {
      await processFiles(Array.from(e.target.files));
      setDefault();
      e.target.value = "";
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processFiles(Array.from(e.dataTransfer.files));
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

  const modelDisplayNames = {
    "llama3.1:8b": "LLaMA 3.1 8B",
    "adrienbrault/saul-instruct-v1:Q8_0": "Saul Instruct",
    "OussamaELALLAM/MedExpert:latest": "MedExpert",
  };

  const displayName = modelDisplayNames[model] || model;

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
      <img src="/uploadIcon.svg" alt="Upload icon" />
      <div className="alpha-upload-hint">
        <strong>Upload documents</strong><br /> <span className="pdf-accent">PDF</span>,{" "}
        <span className="docx-accent">DOCX</span>, or{" "}
        <span className="txt-accent">TXT</span> Files{" "}<br />
        <span className="dim">(max 20MB each)</span>
        <br />
      </div>

      <input
        type="file"
        accept=".pdf,.docx,.txt,.json"
        multiple
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
        title={`Selected Model: ${displayName}`}
        options={[
          { label: "LLaMA 3.1 8B", value: "llama3.1:8b" },
          { label: "Saul Instruct", value: "adrienbrault/saul-instruct-v1:Q8_0" },
          { label: "MedExpert", value: "OussamaELALLAM/MedExpert:latest" },
        ]}
        onSelect={(option) => {
          setModel(option.value);
          console.log("Selected:", option.value);
        }}
      />
    </div>
  );
}