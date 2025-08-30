#!/usr/bin/env python3
import requests
import time

def test_helpful_chatbot():
    print("🚀 TESTING ENHANCED HELPFUL CHATBOT")
    print("=" * 60)
    
    test_queries = [
        "How do I return an item?",
        "My order is late", 
        "Payment failed",
        "When will my package arrive?",
        "Talk to human"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        print("-" * 40)
        
        try:
            response = requests.post('http://localhost:5000/chat', json={
                'query': query,
                'session_id': f'helpful_test_{i}'
            })
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get('response', 'No response')
                
                print(f"📝 Response ({len(bot_response)} chars):")
                print(bot_response)
                
                # Check if it's truly helpful (not generic)
                if any(generic in bot_response.lower() for generic in [
                    "i can help you with that", 
                    "can you provide more details",
                    "need more details"
                ]):
                    print("⚠️  Still showing generic response")
                else:
                    print("✅ Helpful, specific response!")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test complete!")

if __name__ == "__main__":
    time.sleep(3)  # Give Flask time to start
    test_helpful_chatbot()
