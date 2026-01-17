"""
Test script for Knowledge Base Integration
This script tests the knowledge base functionality by:
1. Adding test entries to the knowledge base
2. Verifying retrieval of relevant entries
3. Testing the /ask endpoint with knowledge base integration
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_add_knowledge():
    """Test adding knowledge base entries"""
    print("\n=== Testing Knowledge Base Addition ===")
    
    test_entries = {
        "items": [
            {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "source": "User correction"
            },
            {
                "question": "What is photosynthesis?",
                "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                "source": "User correction"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare wrote Romeo and Juliet.",
                "source": "User correction"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/knowledge_base", json=test_entries)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.status_code == 200

def test_get_knowledge():
    """Test retrieving knowledge base entries"""
    print("\n=== Testing Knowledge Base Retrieval ===")
    
    response = requests.get(f"{BASE_URL}/knowledge_base")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        entries = response.json()
        print(f"Found {len(entries)} knowledge base entries:")
        for entry in entries[:5]:  # Show first 5
            print(f"  - Q: {entry['question'][:50]}...")
            print(f"    A: {entry['answer'][:50]}...")
    
    return response.status_code == 200

def test_ask_with_kb():
    """Test asking a question that should match knowledge base"""
    print("\n=== Testing /ask with Knowledge Base Integration ===")
    
    test_question = {
        "selected_text": "This is a test document about France.",
        "question": "What is the capital city of France?",
        "model": "llama3.1:8b",
        "auth_token": None
    }
    
    print(f"Question: {test_question['question']}")
    print("Sending request...")
    
    response = requests.post(f"{BASE_URL}/ask", json=test_question, stream=True)
    
    if response.status_code == 200:
        print("Response received (streaming):")
        full_response = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                full_response += chunk
                # Print in real-time (optional)
                # print(chunk, end='', flush=True)
        
        print("\n--- Full Response ---")
        # Remove metadata markers
        clean_response = full_response
        clean_response = clean_response.split("__CONTEXT__")[0] if "__CONTEXT__" in clean_response else clean_response
        clean_response = clean_response.split("__ENTRY_ID__")[0] if "__ENTRY_ID__" in clean_response else clean_response
        print(clean_response[:500])  # Print first 500 chars
        
        # Check if knowledge base was used
        if "Paris" in full_response:
            print("\n✓ Knowledge base entry likely used (answer contains 'Paris')")
        else:
            print("\n⚠ Knowledge base entry may not have been used")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    return response.status_code == 200

def test_submit_correction():
    """Test submitting a correction"""
    print("\n=== Testing Correction Submission ===")
    
    # First, ask a question to get a chat_id
    ask_response = requests.post(f"{BASE_URL}/ask", json={
        "selected_text": "Test text",
        "question": "What is 2+2?",
        "model": "llama3.1:8b",
        "auth_token": None
    }, stream=True)
    
    # Extract entry ID from response
    full_response = ""
    for chunk in ask_response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            full_response += chunk
    
    # Try to extract entry ID
    import re
    match = re.search(r'__ENTRY_ID__(\d+)__', full_response)
    
    if match:
        entry_id = int(match.group(1))
        print(f"Got entry ID: {entry_id}")
        
        # Submit correction
        correction = {
            "chat_id": entry_id,
            "corrected_answer": "2+2 equals 4. This is a basic arithmetic fact.",
            "comment": "Test correction"
        }
        
        response = requests.post(f"{BASE_URL}/submit_correction", json=correction)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        return response.status_code == 200
    else:
        print("Could not extract entry ID from response")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Knowledge Base Integration Test Suite")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/test")
        print(f"\n✓ Backend is running: {response.json()['message']}")
    except Exception as e:
        print(f"\n✗ Backend is not running: {e}")
        print("Please start the backend with: python main.py")
        return
    
    # Run tests
    results = {
        "Add Knowledge": test_add_knowledge(),
        "Get Knowledge": test_get_knowledge(),
        "Ask with KB": test_ask_with_kb(),
        "Submit Correction": test_submit_correction()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
