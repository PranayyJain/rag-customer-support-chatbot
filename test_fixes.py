#!/usr/bin/env python3
import requests
import time

def test_critical_fixes():
    print("🔧 TESTING CRITICAL FIXES")
    print("=" * 60)
    
    base_url = 'http://localhost:5000'
    
    # Test 1: Return context flow
    print("\n1️⃣ TESTING RETURN CONTEXT FLOW:")
    print("-" * 40)
    
    session = 'test_return_context'
    
    # Step 1: Ask about returns
    r1 = requests.post(f'{base_url}/chat', json={'query': 'Return item', 'session_id': session})
    print("User: Return item")
    print(f"Bot: {r1.json()['response'][:100]}...")
    
    # Step 2: Provide order ID (should trigger return processing, not tracking)
    r2 = requests.post(f'{base_url}/chat', json={'query': 'order id is 4567', 'session_id': session})
    response2 = r2.json()['response']
    print("\nUser: order id is 4567")
    print(f"Bot: {response2[:150]}...")
    
    if "return" in response2.lower() and "process" in response2.lower():
        print("✅ RETURN CONTEXT: FIXED!")
    else:
        print("❌ RETURN CONTEXT: Still broken")
    
    # Test 2: Ticket creation
    print("\n\n2️⃣ TESTING TICKET CREATION:")
    print("-" * 40)
    
    session2 = 'test_ticket'
    
    # Step 1: Request human support
    r3 = requests.post(f'{base_url}/chat', json={'query': 'Talk to human', 'session_id': session2})
    print("User: Talk to human")
    print(f"Bot: {r3.json()['response']}")
    
    # Step 2: Confirm ticket creation
    r4 = requests.post(f'{base_url}/chat', json={'query': 'yes please', 'session_id': session2})
    response4 = r4.json()['response']
    print("\nUser: yes please")
    print(f"Bot: {response4}")
    
    if "TKT" in response4 and "created" in response4.lower():
        print("✅ TICKET CREATION: FIXED!")
    else:
        print("❌ TICKET CREATION: Still broken")
    
    # Test 3: Payment failure specificity
    print("\n\n3️⃣ TESTING PAYMENT FAILURE HANDLING:")
    print("-" * 40)
    
    r5 = requests.post(f'{base_url}/chat', json={'query': 'payment failed', 'session_id': 'test_payment'})
    response5 = r5.json()['response']
    print("User: payment failed")
    print(f"Bot: {response5[:200]}...")
    
    if "card details" in response5.lower() and "check" in response5.lower():
        print("✅ PAYMENT SPECIFICITY: FIXED!")
    else:
        print("❌ PAYMENT SPECIFICITY: Still generic")
    
    print("\n" + "=" * 60)
    print("🎯 Fix testing complete!")

if __name__ == "__main__":
    time.sleep(4)  # Give Flask time to start
    test_critical_fixes()