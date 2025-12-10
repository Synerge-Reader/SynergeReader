# Quick Start Guide: Citations & Answer Corrections

## 🎯 Feature 1: Citations in File Metadata

### What It Does
Automatically attach citation information to uploaded documents for better traceability and academic integrity.

### How to Use

#### Step 1: Upload a Document
1. Navigate to the file upload area
2. You'll see citation input fields:
   - **Title**: Document title
   - **Author**: Author name(s)
   - **Source/Publisher**: Where it was published
   - **Date**: Publication date (YYYY-MM-DD format)
   - **DOI or URL**: Digital identifier or web link

#### Step 2: Fill Citation Information (Optional)
```
Title: "The Impact of AI on Healthcare"
Author: "Dr. Jane Smith"
Source/Publisher: "Medical Journal of AI"
Date: "2024-03-15"
DOI or URL: "https://doi.org/10.1234/example"
```

#### Step 3: Upload File
- Click "Browse Files" or drag and drop
- Citation information is automatically saved with the document

#### Step 4: View Citations
- Citations appear in the Document Preview section
- Displayed in a formatted box above the document text
- Format: "Title" by Author (Date) - Source [DOI/URL]

### Example Output
```
Citation: "The Impact of AI on Healthcare" by Dr. Jane Smith (2024-03-15) - Medical Journal of AI [https://doi.org/10.1234/example]
```

---

## 🎯 Feature 2: Answer Correction System

### What It Does
- Mark answers as correct or incorrect
- Provide corrected answers
- Build a knowledge base that improves future responses
- System learns from your corrections

### How to Use

#### Scenario A: Answer is Correct ✓

1. **Ask a question** about your document
2. **Review the answer** from the AI
3. **Click** "Provide Feedback / Correct Answer" button
4. **Click** "✓ Mark as Correct" in the modal
5. **Done!** Answer is saved to knowledge base

**What happens:**
- Answer is marked as verified
- Saved to knowledge base
- Future similar questions will reference this verified answer

#### Scenario B: Answer is Incorrect ✗

1. **Ask a question** about your document
2. **Notice the answer is wrong**
3. **Click** "Provide Feedback / Correct Answer" button
4. **Type the correct answer** in the text box
5. **Click** "Submit Correction"
6. **Done!** Correction is saved

**What happens:**
- Original answer is replaced with your correction
- Correction saved to knowledge base
- Future similar questions will use your corrected answer
- AI learns from your feedback

### Example Workflow

#### Initial Question
```
Q: What is the capital of France?
A: The capital of France is Lyon.  ← WRONG!
```

#### Correction Process
1. Click "Provide Feedback / Correct Answer"
2. Modal shows:
   - Original Question: "What is the capital of France?"
   - Original Answer: "The capital of France is Lyon."
3. Type correction: "The capital of France is Paris."
4. Click "Submit Correction"

#### Future Questions
```
Q: What city is the capital of France?
A: The capital of France is Paris.  ← CORRECT! (Uses knowledge base)
```

---

## 🔄 How Knowledge Base Integration Works

### The Learning Loop

```
1. User asks question
   ↓
2. System checks knowledge base for similar questions
   ↓
3. If found: Include verified answer in AI prompt
   ↓
4. AI generates better response using verified knowledge
   ↓
5. User confirms or corrects answer
   ↓
6. Correction saved to knowledge base
   ↓
7. Loop continues (system gets smarter!)
```

### Behind the Scenes

When you ask a question, the system:
1. **Searches** knowledge base for similar questions
2. **Retrieves** top 2 most relevant verified answers
3. **Includes** them in the AI prompt as context
4. **Generates** answer using both documents AND knowledge base
5. **Learns** from your feedback

---

## 📊 Viewing Your Knowledge Base

### API Endpoint
```
GET http://localhost:5000/knowledge_base
```

### Response Format
```json
[
  {
    "id": 1,
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "created_at": "2025-12-02T21:00:00",
    "chat_history_id": 42
  }
]
```

### Using the Test Script
```bash
cd c:\Users\Lenovo\Documents\SynergeReader
python test_knowledge_base.py
```

This will:
- Add test knowledge entries
- Retrieve all entries
- Test asking questions with KB integration
- Test submitting corrections

---

## 💡 Best Practices

### For Citations
- ✅ Fill in as much citation info as possible
- ✅ Use consistent date format (YYYY-MM-DD)
- ✅ Include DOI when available (more reliable than URLs)
- ✅ Double-check author names and titles
- ⚠️ Citations are optional but recommended for academic work

### For Corrections
- ✅ Be specific and accurate in corrections
- ✅ Mark correct answers to build verified knowledge
- ✅ Provide context in corrections when helpful
- ✅ Review AI answers before marking as correct
- ⚠️ Corrections are permanent (stored in knowledge base)

### For Knowledge Base
- ✅ Build up knowledge base over time
- ✅ Correct answers as you find errors
- ✅ Mark good answers as correct
- ✅ Use consistent terminology
- ⚠️ Knowledge base improves with more entries

---

## 🐛 Troubleshooting

### Citations Not Showing
- **Check**: Did you fill in at least one citation field?
- **Check**: Is the document uploaded successfully?
- **Check**: Refresh the page and re-upload

### Correction Not Saving
- **Check**: Is the backend running?
- **Check**: Did you provide a corrected answer?
- **Check**: Check browser console for errors
- **Check**: Verify database connection

### Knowledge Base Not Working
- **Check**: Are there entries in the knowledge base? (Use GET /knowledge_base)
- **Check**: Is your question similar to existing entries?
- **Check**: Backend logs for errors
- **Try**: Add test entries using test_knowledge_base.py

### Backend Issues
```bash
# Check if backend is running
curl http://localhost:5000/test

# Expected response:
{"message": "SynergeReader API is running successfully!"}
```

---

## 🔧 Advanced Usage

### Manually Add Knowledge Entries

```python
import requests

entries = {
    "items": [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI...",
            "source": "Expert review"
        }
    ]
}

response = requests.post(
    "http://localhost:5000/knowledge_base",
    json=entries
)
print(response.json())
```

### Bulk Import Citations

Modify `FileUpload.js` to read citation data from a CSV or JSON file and auto-populate fields.

### Export Knowledge Base

```python
import requests
import json

response = requests.get("http://localhost:5000/knowledge_base")
knowledge = response.json()

with open("knowledge_export.json", "w") as f:
    json.dump(knowledge, f, indent=2)

print(f"Exported {len(knowledge)} entries")
```

---

## 📚 Additional Resources

- **Implementation Report**: `IMPLEMENTATION_REPORT.md` - Full technical details
- **Test Script**: `test_knowledge_base.py` - Automated testing
- **Backend Code**: `synerge-reader-backend/main.py` - API implementation
- **Frontend Components**:
  - `FileUpload.js` - Citation inputs
  - `TextPreview.js` - Citation display
  - `CorrectionModal.jsx` - Correction interface
  - `GridApp.jsx` - Main integration

---

## 🎉 Summary

You now have:
1. ✅ **Citation tracking** for all uploaded documents
2. ✅ **Answer correction** system with user feedback
3. ✅ **Knowledge base** that learns from corrections
4. ✅ **Improved AI responses** using verified knowledge

The system gets smarter with every correction you make! 🚀
