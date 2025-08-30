#!/usr/bin/env python3
import requests
import time
import json

def test_more_details():
    session_id = 'test_session_123'
    base_url = 'http://localhost:5000'
    
    print("🧪 Testing More Details Functionality")
    print("=" * 50)
    
    # Step 1: Initial query
    print("Step 1: Asking about returns...")
    response1 = requests.post(f'{base_url}/chat', json={
        'query': 'return item',
        'session_id': session_id
    })
    
    if response1.status_code == 200:
        data1 = response1.json()
        print(f"✅ Response: {data1['response']}")
        print(f"📊 Intent: {data1.get('intent', 'Unknown')}")
        print()
    else:
        print(f"❌ Error: {response1.status_code}")
        return
    
    # Step 2: Ask for more details
    print("Step 2: Asking for more details...")
    response2 = requests.post(f'{base_url}/chat', json={
        'query': 'More details',
        'session_id': session_id
    })
    
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"📚 More Details Response:")
        print(f"   {data2['response']}")
        print(f"📊 Intent: {data2.get('intent', 'Unknown')}")
        print()
        
        if "specialized in helping" in data2['response']:
            print("❌ PROBLEM: Getting generic fallback instead of stored content")
        elif "Detailed Information" in data2['response']:
            print("✅ SUCCESS: Getting stored detailed content!")
        else:
            print("❓ UNKNOWN: Unexpected response format")
    else:
        print(f"❌ Error: {response2.status_code}")

if __name__ == "__main__":
    test_more_details()
