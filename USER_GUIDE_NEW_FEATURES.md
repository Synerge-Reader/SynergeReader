# How to Use Citation and Correction Features

## 1. Citation Metadata
1.  **Upload a Document**:
    -   Click "Browse Files" or drag & drop a file.
    -   **Before** selecting the file, fill in the "Citation Metadata" fields (Title, Author, Source, Date, DOI) if desired.
    -   These fields apply to the batch of files being uploaded.
2.  **View Citations**:
    -   After upload, the citation information will appear in the "Document Previews" section above the document text.
    -   The metadata is also stored in the database for future retrieval.

## 2. Answer Correction & Knowledge Base
1.  **Ask a Question**:
    -   Select text and ask a question as usual.
2.  **Provide Feedback**:
    -   Below the AI's answer, click the **"Provide Feedback / Correct Answer"** button.
3.  **Submit Correction**:
    -   **Mark as Correct**: Click "✓ Mark as Correct" if the answer is accurate. This saves it to the Knowledge Base as a verified answer.
    -   **Submit Correction**: If the answer is wrong, type the correct answer in the text box and click "Submit Correction". This saves the correction to the Knowledge Base.
4.  **Knowledge Base**:
    -   All corrections are stored in the `knowledge_base` table.
    -   Future improvements can use this knowledge base to improve AI responses (RAG).
