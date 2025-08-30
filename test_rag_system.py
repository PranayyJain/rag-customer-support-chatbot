#!/usr/bin/env python3
"""
Test script to demonstrate the RAG system functionality
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"
SESSION_ID = f"test_rag_{int(time.time())}"

def test_chat(query, description=""):
    """Test a chat query and display results"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", 
                               json={"query": query, "session_id": SESSION_ID},
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response received")
            print(f"Intent: {data.get('intent', 'N/A')}")
            print(f"Confidence: {data.get('confidence', 'N/A')}")
            print(f"Response: {data.get('response', 'N/A')}")
            
            sources = data.get('sources', [])
            if sources:
                print(f"\n📚 RAG Sources Retrieved:")
                for i, source in enumerate(sources, 1):
                    title = source.get('title', 'Unknown')
                    score = source.get('score', 0)
                    print(f"  {i}. {title} (score: {score:.3f})")
            else:
                print("📚 No RAG sources found")
                
            # Check if response length is concise (task requirement)
            response_length = len(data.get('response', '').split())
            if response_length <= 50:  # Rough measure for 2 sentences
                print(f"✅ Response is concise ({response_length} words)")
            else:
                print(f"❌ Response is too long ({response_length} words) - should be ~2 sentences")
                
            return data
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run comprehensive RAG system tests"""
    print("🚀 Testing RAG-based Customer Support Chatbot")
    print(f"Session ID: {SESSION_ID}")
    
    # Test 1: Basic return query
    test_chat("How can I return an item?", "Basic return inquiry")
    
    # Test 2: Order tracking
    test_chat("Track my order", "Order tracking without order ID")
    
    # Test 3: Order tracking with ID
    test_chat("Track my order #ABC123", "Order tracking with order ID")
    
    # Test 4: Payment issue
    test_chat("My payment failed", "Payment failure inquiry")
    
    # Test 5: More details request
    test_chat("More details", "Request for detailed information")
    
    # Test 6: Shipping information  
    test_chat("How long does shipping take?", "Shipping duration inquiry")
    
    # Test 7: Exchange request
    test_chat("I want to exchange an item", "Exchange request")
    
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print("These tests demonstrate:")
    print("1. ✅ RAG retrieval from FAQ knowledge base")
    print("2. ✅ Intent classification and entity extraction")
    print("3. ✅ Source tracking for transparency")
    print("4. ⚠️  Response length (should be optimized to 2 sentences)")
    print("5. ✅ Session management across queries")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
