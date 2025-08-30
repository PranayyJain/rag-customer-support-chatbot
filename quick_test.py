import time
import requests

time.sleep(3)

print("🧪 QUICK TESTS")
print("=" * 40)

# Test 1: Return context
print("\n1️⃣ Return Context:")
r1 = requests.post('http://localhost:5000/chat', json={'query': 'Return item', 'session_id': 'test1'})
r2 = requests.post('http://localhost:5000/chat', json={'query': 'order id is 4567', 'session_id': 'test1'})
result1 = r2.json()['response']
print("RESULT:", "✅ FIXED" if "return" in result1.lower() and "process" in result1.lower() else "❌ BROKEN")

# Test 2: Ticket creation
print("\n2️⃣ Ticket Creation:")
r3 = requests.post('http://localhost:5000/chat', json={'query': 'Talk to human', 'session_id': 'test2'})
r4 = requests.post('http://localhost:5000/chat', json={'query': 'yes please', 'session_id': 'test2'})
if r4.status_code == 200:
    result2 = r4.json().get('response', '')
    print("RESULT:", "✅ FIXED" if "TKT" in result2 else "❌ BROKEN")
    print("Response:", result2[:80])
else:
    print("RESULT: ❌ ERROR (Status:", r4.status_code, ")")

# Test 3: Payment specificity
print("\n3️⃣ Payment Specificity:")
r5 = requests.post('http://localhost:5000/chat', json={'query': 'payment failed', 'session_id': 'test3'})
result3 = r5.json()['response']
print("RESULT:", "✅ FIXED" if "card details" in result3.lower() else "❌ BROKEN")

print("\n" + "=" * 40)
