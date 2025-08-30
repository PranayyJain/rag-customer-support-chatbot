import os
import json
import logging
import datetime
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import faiss
import cohere
from groq import Groq
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sqlite3
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Paths configurable via environment for deployment (e.g., Render disk)
DB_PATH = os.getenv('DB_PATH', 'conversations.db')
ANALYTICS_PATH = os.getenv('ANALYTICS_PATH', 'analytics.json')

# Ensure directories/files exist for DB and analytics paths
try:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    analytics_dir = os.path.dirname(ANALYTICS_PATH)
    if analytics_dir:
        os.makedirs(analytics_dir, exist_ok=True)
    if not os.path.exists(ANALYTICS_PATH):
        with open(ANALYTICS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
except Exception as _e:
    logger.warning(f"Could not prepare data paths: {DB_PATH}, {ANALYTICS_PATH}: {_e}")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize APIs with error handling
try:
    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
    logger.info("API clients initialized successfully")
except Exception as e:
    logger.warning(f"API clients not initialized: {e}")
    groq_client = None
    cohere_client = None

@dataclass
class CustomerQuery:
    query: str
    session_id: str
    timestamp: datetime.datetime
    intent: Optional[str] = None
    entities: Optional[Dict] = None
    confidence: Optional[float] = None

@dataclass
class KnowledgeDocument:
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    embedding: Optional[np.ndarray] = None

class IntentClassifier:
    def __init__(self):
        self.intents = {
            'order_status': ['order', 'status', 'tracking', 'delivery', 'shipped', 'delivered'],
            'returns_refunds': ['return', 'refund', 'exchange', 'money back', 'cancel order'],
            'payment_billing': ['payment', 'billing', 'charge', 'credit card', 'payment failed'],
            'product_inquiry': ['product', 'item', 'specification', 'availability', 'stock'],
            'technical_support': ['not working', 'broken', 'defective', 'issue', 'problem', 'bug'],
            'account_management': ['account', 'profile', 'password', 'login', 'registration'],
            'shipping_delivery': ['shipping', 'delivery', 'address', 'location', 'fast delivery'],
            'general_complaint': ['complaint', 'dissatisfied', 'poor service', 'bad experience']
        }
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Enhanced intent classification with embeddings similarity"""
        try:
            if cohere_client is None:
                return self._keyword_based_classification(query)
            
            # Use embeddings similarity for classification (since classify API is deprecated)
            query_embedding = cohere_client.embed(texts=[query], model="embed-multilingual-v3.0", input_type="search_query").embeddings[0]
            
            # Define intent examples with embeddings
            intent_examples = {
                "order_status": ["track my package", "where is my order", "order status", "delivery update"],
                "payment_billing": ["payment failed", "billing issue", "charge problem", "payment error"],
                "returns_refunds": ["return product", "refund request", "exchange item", "return policy"],
                "account_management": ["forgot password", "account help", "login issue", "update profile"],
                "technical_support": ["app not working", "website error", "technical problem", "bug report"],
                "general_complaint": ["talk to human", "speak to agent", "customer service", "human support"],
                "shipping_delivery": ["when will it arrive", "shipping info", "delivery time", "shipping cost"]
            }
            
            best_intent = "general_inquiry"
            best_score = 0.0
            
            for intent, examples in intent_examples.items():
                # Get embeddings for examples
                example_embeddings = cohere_client.embed(texts=examples, model="embed-multilingual-v3.0", input_type="search_document").embeddings
                
                # Calculate similarity with query
                similarities = [np.dot(query_embedding, example_emb) for example_emb in example_embeddings]
                max_similarity = max(similarities)
                
                if max_similarity > best_score:
                    best_score = max_similarity
                    best_intent = intent
            
            # Normalize confidence score
            confidence = min(best_score * 1.5, 1.0)
            return best_intent, confidence
            
        except Exception as e:
            logger.warning(f"Embedding classification failed, using fallback: {e}")
            return self._keyword_based_classification(query)
    
    def _keyword_based_classification(self, query: str) -> Tuple[str, float]:
        """Fallback keyword-based intent classification"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, keywords in self.intents.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score / len(keywords)
        
        if not any(intent_scores.values()):
            return 'general_inquiry', 0.5
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        return best_intent, min(confidence * 2, 1.0)

class EntityExtractor:
    def __init__(self):
        self.patterns = {
            'order_number': r'(?:order|order number|tracking)\s*[#:]?\s*([A-Z0-9]{6,15})',
            'product_name': r'(?:product|item)\s+([A-Za-z\s]+)',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'amount': r'\$?(\d+\.?\d*)',
            'date': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'
        }
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {}
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        return entities

class KnowledgeBase:
    def __init__(self):
        self.documents = []
        self.index = None
        self.embeddings = []
        self.load_knowledge_base()
        self.build_faiss_index()
    
    def load_knowledge_base(self):
        """Load predefined ecommerce knowledge base"""
        knowledge_data = [
            {
                "id": "order_001",
                "title": "How to track your order",
                "content": "To track your order: 1) Go to 'My Orders' section 2) Enter your order number 3) View real-time tracking status. You'll receive SMS/email updates automatically.",
                "category": "order_status",
                "tags": ["tracking", "order", "status"]
            },
            {
                "id": "return_001", 
                "title": "Return and Refund Policy",
                "content": "You can return items within 30 days of delivery. Items must be unused and in original packaging. Refund processing takes 5-7 business days after we receive the returned item.",
                "category": "returns_refunds",
                "tags": ["return", "refund", "policy"]
            },
            {
                "id": "payment_001",
                "title": "Payment Methods and Issues",
                "content": "We accept credit cards, debit cards, UPI, and digital wallets. If payment fails, check your card details, available balance, or try a different payment method.",
                "category": "payment_billing",
                "tags": ["payment", "billing", "methods"]
            },
            {
                "id": "shipping_001",
                "title": "Shipping and Delivery Information", 
                "content": "Standard delivery: 3-5 business days. Express delivery: 1-2 business days. Free shipping on orders above $50. Delivery charges vary by location.",
                "category": "shipping_delivery",
                "tags": ["shipping", "delivery", "charges"]
            },
            {
                "id": "account_001",
                "title": "Account Management",
                "content": "To manage your account: Update profile information, change password, view order history, manage addresses, and set preferences in the Account Settings section.",
                "category": "account_management", 
                "tags": ["account", "profile", "settings"]
            },
            {
                "id": "product_001",
                "title": "Product Information and Availability",
                "content": "Check product availability on the product page. Out of stock items show 'Notify Me' option. Product specifications, reviews, and images are available on individual product pages.",
                "category": "product_inquiry",
                "tags": ["product", "availability", "specifications"]
            },
            {
                "id": "tech_001",
                "title": "Technical Support and Issues",
                "content": "For technical issues: 1) Clear browser cache 2) Check internet connection 3) Try different browser 4) Contact support if problem persists. Our tech team is available 24/7.",
                "category": "technical_support",
                "tags": ["technical", "support", "issues"]
            },
            {
                "id": "complaint_001",
                "title": "Filing Complaints and Feedback",
                "content": "To file a complaint: Use the 'Contact Us' form, email support@company.com, or call our helpline. Include order details and issue description. We respond within 24 hours.",
                "category": "general_complaint",
                "tags": ["complaint", "feedback", "support"]
            }
        ]
        
        for doc_data in knowledge_data:
            doc = KnowledgeDocument(**doc_data)
            self.documents.append(doc)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Cohere API"""
        if cohere_client is None:
            logger.info("Cohere client not available, using TF-IDF fallback")
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=384)
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray().tolist()
        
        try:
            response = cohere_client.embed(texts=texts, model='embed-english-v2.0')
            return response.embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=384)
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray().tolist()
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        texts = [doc.content for doc in self.documents]
        embeddings = self.get_embeddings(texts)
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            # Store embeddings
            for i, doc in enumerate(self.documents):
                doc.embedding = embeddings_array[i]
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[KnowledgeDocument, float]]:
        """Search for relevant documents"""
        if not self.index:
            return []
        
        try:
            query_embedding = self.get_embeddings([query])[0]
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []

class ConversationManager:
    def __init__(self):
        self.sessions = {}
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for conversation history"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                bot_response TEXT,
                intent TEXT,
                confidence REAL,
                timestamp DATETIME,
                entities TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, user_query: str, bot_response: str, 
                         intent: str, confidence: float, entities: Dict):
        """Save conversation to database and analytics"""
        # Save to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, user_query, bot_response, intent, confidence, timestamp, entities)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_query, bot_response, intent, confidence, 
              datetime.datetime.now(), json.dumps(entities)))
        conn.commit()
        conn.close()
        
        # Log analytics
        self._log_analytics(session_id, user_query, intent, confidence, entities)
    
    def _log_analytics(self, session_id: str, user_query: str, intent: str, 
                       confidence: float, entities: Dict, sources: List = None):
        """Log analytics data to JSON file with RAG sources"""
        try:
            analytics_data = {
                "timestamp": time.time(),
                "session_id": session_id,
                "user_message": user_query,
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "sources": sources or [],  # RAG sources for proof
                "response_time": time.time()  # Will be updated in process_query
            }
            
            # Load existing analytics or create new
            analytics_file = ANALYTICS_PATH
            if os.path.exists(analytics_file):
                with open(analytics_file, "r") as f:
                    analytics_log = json.load(f)
            else:
                analytics_log = []
            
            analytics_log.append(analytics_data)
            
            # Save updated analytics
            with open(analytics_file, "w") as f:
                json.dump(analytics_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log analytics: {e}")
    
    def get_conversation_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_query, bot_response, intent, timestamp 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'user_query': row[0],
                'bot_response': row[1], 
                'intent': row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return list(reversed(history))

class TicketSystem:
    def __init__(self):
        self.init_ticket_database()
    
    def init_ticket_database(self):
        """Initialize SQLite database for tickets"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE,
                session_id TEXT,
                customer_name TEXT,
                customer_email TEXT,
                customer_phone TEXT,
                issue_description TEXT,
                issue_category TEXT,
                status TEXT DEFAULT 'open',
                priority TEXT DEFAULT 'medium',
                created_at DATETIME,
                conversation_history TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_ticket_id(self) -> str:
        """Generate unique ticket ID"""
        import random
        import string
        return 'TKT' + ''.join(random.choices(string.digits, k=6))
    
    def create_ticket(self, session_id: str, customer_info: Dict, issue_description: str, 
                     issue_category: str, conversation_history: List) -> str:
        """Create a new support ticket"""
        ticket_id = self.generate_ticket_id()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO tickets 
                (ticket_id, session_id, customer_name, customer_email, customer_phone,
                 issue_description, issue_category, created_at, conversation_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticket_id, session_id, 
                customer_info.get('name', ''),
                customer_info.get('email', ''),
                customer_info.get('phone', ''),
                issue_description, issue_category,
                datetime.datetime.now(),
                json.dumps(conversation_history)
            ))
            conn.commit()
            return ticket_id
        except sqlite3.IntegrityError:
            # If ticket ID already exists, generate a new one
            conn.close()
            return self.create_ticket(session_id, customer_info, issue_description, issue_category, conversation_history)
        finally:
            conn.close()

class ConversationState:
    def __init__(self):
        self.collecting_info = False
        self.info_type = None
        self.collected_info = {}
        self.interaction_count = 0
        self.issue_resolved = False
        self.awaiting_ticket = False
        self.awaiting_ticket_confirmation = False
        self.last_intent = None
        self.last_retrieved = ""
        self.last_sources = []

class CustomerSupportChatbot:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.knowledge_base = KnowledgeBase()  # Keep for fallback
        self.conversation_manager = ConversationManager()
        self.ticket_system = TicketSystem()
        self.session_states = {}  # Track conversation states per session
        
        # Initialize RAG pipeline
        try:
            self.rag_pipeline = RAGPipeline(cohere_client)
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.warning(f"RAG pipeline initialization failed: {e}")
            self.rag_pipeline = None
        
        # Define non-ecommerce keywords to filter out
        self.non_ecommerce_keywords = [
            'weather', 'news', 'sports', 'politics', 'cooking', 'recipes',
            'movie', 'music', 'travel', 'health', 'medicine', 'education',
            'programming', 'code', 'math', 'science', 'history'
        ]
        
        # Quick responses for common phrases
        self.quick_responses = {
            'greeting': [
                "Hi there! I'm here to help with your shopping needs. What can I assist you with today?",
                "Hello! How can I help you with your order or account?",
                "Hey! I'm your customer support assistant. What do you need help with?"
            ],
            'thanks': [
                "You're welcome! Anything else I can help with?",
                "Happy to help! Let me know if you need anything else.",
                "Glad I could assist! Any other questions?"
            ],
            'yes': [
                "Great! What can I help you with?",
                "Perfect! Please tell me more.",
                "Awesome! How can I assist?"
            ],
            'no': [
                "No worries! What else can I help you with?",
                "That's fine! Anything else I can assist with?",
                "Understood! Any other questions?"
            ],
            'goodbye': [
                "Thank you for contacting us! Have a great day!",
                "Goodbye! Feel free to reach out anytime.",
                "Take care! We're always here when you need help."
            ]
        }
    
    def is_ecommerce_related(self, query: str) -> bool:
        """Check if query is ecommerce customer support related"""
        query_lower = query.lower()
        
        # Check for non-ecommerce keywords
        if any(keyword in query_lower for keyword in self.non_ecommerce_keywords):
            return False
        
        # Check for ecommerce-related keywords
        ecommerce_keywords = [
            'order', 'delivery', 'shipping', 'return', 'refund', 'payment',
            'product', 'item', 'account', 'billing', 'tracking', 'cancel',
            'exchange', 'warranty', 'support', 'help', 'issue', 'problem'
        ]
        
        if any(keyword in query_lower for keyword in ecommerce_keywords):
            return True
        
        # If no specific keywords found, allow general queries that might be support-related
        question_words = ['what', 'how', 'where', 'when', 'why', 'can', 'help']
        if any(word in query_lower for word in question_words):
            return True
        
        return False
    
    def get_session_state(self, session_id: str) -> ConversationState:
        """Get or create conversation state for session"""
        if session_id not in self.session_states:
            self.session_states[session_id] = ConversationState()
        return self.session_states[session_id]
    
    def detect_quick_response_type(self, query: str) -> str:
        """Detect if query needs a quick response"""
        query_lower = query.lower().strip()
        
        # Greeting patterns
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in query_lower for greeting in greetings) and len(query_lower) < 20:
            return 'greeting'
        
        # Thanks patterns
        thanks = ['thank you', 'thanks', 'thank u', 'thx', 'appreciated', 'grateful']
        if any(thank in query_lower for thank in thanks):
            return 'thanks'
        
        # Yes patterns
        yes_patterns = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'correct', 'right']
        if query_lower in yes_patterns:
            return 'yes'
        
        # No patterns  
        no_patterns = ['no', 'nope', 'nah', 'not really', 'incorrect', 'wrong']
        if query_lower in no_patterns:
            return 'no'
        
        # Goodbye patterns
        goodbye = ['bye', 'goodbye', 'see you', 'farewell', 'have a good day']
        if any(bye in query_lower for bye in goodbye):
            return 'goodbye'
        
        return None
    
    def collect_customer_info(self, query: str, session_state: ConversationState) -> Tuple[str, bool]:
        """Collect customer information for ticket creation"""
        if not session_state.info_type:
            session_state.info_type = 'name'
            return "To better assist you, I'll need some information. What's your full name?", False
        
        if session_state.info_type == 'name':
            session_state.collected_info['name'] = query.strip()
            session_state.info_type = 'email'
            return "Thank you! What's your email address?", False
        
        elif session_state.info_type == 'email':
            # Basic email validation
            if '@' in query and '.' in query:
                session_state.collected_info['email'] = query.strip()
                session_state.info_type = 'phone'
                return "Great! And your phone number?", False
            else:
                return "Please provide a valid email address.", False
        
        elif session_state.info_type == 'phone':
            session_state.collected_info['phone'] = query.strip()
            session_state.collecting_info = False
            session_state.info_type = None
            return "", True  # Info collection complete
        
        return "", False
    
    def should_create_ticket(self, session_state: ConversationState, intent: str) -> bool:
        """Determine if ticket should be created based on conversation flow"""
        # Create ticket if:
        # 1. User has interacted 3+ times without resolution
        # 2. Intent is complex (complaint, technical issue)
        # 3. User explicitly asks for human help
        
        complex_intents = ['general_complaint', 'technical_support', 'account_management']
        
        return (session_state.interaction_count >= 3 and not session_state.issue_resolved) or \
               intent in complex_intents or \
               session_state.awaiting_ticket
    
    def generate_contextual_response(self, query: str, intent: str, entities: Dict, 
                                   relevant_docs: List, session_state: ConversationState, session_id: str) -> str:
        """Generate contextual response using RAG pipeline for concise responses"""
        # Always use RAG pipeline for concise responses instead of Groq
        logger.info("Using RAG pipeline for concise response generation")
        return self._generate_knowledge_base_response(query, intent, entities, relevant_docs, session_id, session_state)
    
    def _generate_knowledge_base_response(self, query: str, intent: str, entities: Dict, 
                                        relevant_docs: List, session_id: str, session_state=None) -> str:
        """Generate responses using RAG pipeline or fallback to knowledge base"""
        
        # Always use RAG pipeline for true concise responses
        if self.rag_pipeline:
            try:
                logger.info(f"Attempting RAG pipeline for query: {query[:50]}...")
                concise_reply, retrieved_text, sources, latency = self.rag_pipeline.rag_response(query)
                logger.info(f"RAG success - retrieved content length: {len(retrieved_text) if retrieved_text else 0}")
                
                # Add contextual follow-up for missing order ID
                if intent in ["order_status", "returns_refunds", "payment_billing", "shipping_delivery"]:
                    if not any(char.isdigit() for char in query):  # crude order_id check
                        concise_reply += "\n\nCan you share your order ID so I can check this for you?"
                
                # Add "More details" option
                if intent in ['order_status', 'returns_refunds', 'payment_billing', 'shipping_delivery']:
                    concise_reply += "\n\n*Need more details? Just ask for 'More details'*"
                
                # Store retrieved text for "More details" requests
                if session_state is None:
                    session_state = self.get_session_state(session_id)
                session_state.last_retrieved = retrieved_text
                session_state.last_sources = sources
                
                # Ensure persistence in chatbot session dictionary
                if session_id not in self.session_states:
                    self.session_states[session_id] = session_state
                self.session_states[session_id].last_retrieved = retrieved_text
                self.session_states[session_id].last_sources = sources
                
                return concise_reply
            except Exception as e:
                logger.error(f"RAG pipeline failed: {e}")
                pass  # Continue to fallback logic
        
        # Fallback to original knowledge base logic
        logger.info("Using fallback knowledge base logic")
        session_history = self.conversation_manager.get_conversation_history(session_id, limit=5)
        recent_intents = [h.get('intent', '') for h in session_history if h.get('intent')]
        
        # Progressive responses based on session context
        if intent in recent_intents[-2:]:  # Asked same intent recently
            return self._get_progressive_response(intent, entities, session_id)
        
        # Generate short, actionable first response
        short_response = self._get_short_response(intent, entities, relevant_docs)
        
        # Store content for "More details" even in fallback
        logger.info(f"Fallback storage - session_state exists: {session_state is not None}, intent: {intent}")
        if session_state:
            if session_id not in self.session_states:
                self.session_states[session_id] = session_state
            
            # Always store some content based on intent for more details
            if intent == "returns_refunds":
                detailed_content = "You can return items within 30 days of delivery. Items must be unused and in original packaging. To initiate a return: 1) Go to 'My Orders' section 2) Select the order and item 3) Choose 'Return Item' 4) Print the return label 5) Package and ship back. Refund processing takes 5-7 business days after we receive the returned item. You can track return status in your account."
            elif intent == "order_status":
                detailed_content = "To track your order: 1) Go to 'My Orders' section 2) Enter your order number 3) View real-time tracking status. You'll receive SMS/email updates automatically. If order shows as 'Processing', it's being prepared for shipment. 'Shipped' means it's on the way with tracking available. 'Out for delivery' means it will arrive today."
            elif intent == "payment_billing":
                detailed_content = "We accept credit cards, debit cards, UPI, and digital wallets. If payment fails, check: 1) Card details are correct 2) Sufficient balance available 3) Card hasn't expired 4) Billing address matches. Try a different payment method if issues persist. For refunds, processing takes 5-7 business days to reflect in your account."
            elif intent == "shipping_delivery":
                detailed_content = "We offer multiple shipping options: Standard (3-5 days), Express (1-2 days), and Same-day (select cities). Free shipping on orders above $50. Track your package using the tracking number sent to your email. If delivery fails, we'll attempt redelivery or hold at nearest pickup point."
            else:
                detailed_content = "Here are more details about your inquiry. For specific assistance, please contact our support team or check the relevant section in your account."
            
            self.session_states[session_id].last_retrieved = detailed_content
            self.session_states[session_id].last_sources = [f"{intent.replace('_', ' ').title()} Information"]
            logger.info(f"✅ STORED {intent} content for session {session_id}: {len(detailed_content)} chars")
            logger.info(f"✅ Session states now contains: {list(self.session_states.keys())}")
        else:
            logger.error(f"❌ session_state is None - cannot store content for session {session_id}")
        
        # Add "More details" option for certain intents
        if intent in ['order_status', 'returns_refunds', 'payment_billing', 'shipping_delivery']:
            short_response += "\n\n*Need more details? Just ask for 'More details'*"
        
        return short_response
    
    def _get_short_response(self, intent: str, entities: Dict, relevant_docs: List) -> str:
        """Generate short 1-2 sentence responses with action focus"""
        
        if intent == 'order_status':
            if 'order_number' in entities and entities['order_number']:
                order_num = entities['order_number'][0]
                return f"📦 I'll check order #{order_num} for you. It should be on track for delivery within 3-5 business days."
            else:
                return "📦 I can help track your order! Please share your order number."
        
        elif intent == 'returns_refunds':
            if 'order_number' in entities and entities['order_number']:
                order_num = entities['order_number'][0]
                return f"🔄 I can help return items from order #{order_num}. You have 30 days from delivery to return unused items."
            else:
                return "🔄 I can help with returns! Share your order number and I'll get the process started."
        
        elif intent == 'payment_billing':
            return "💳 Payment issues can be frustrating. Let me help you resolve this quickly."
        
        elif intent == 'shipping_delivery':
            return "🚚 Standard shipping takes 3-5 business days, express is 1-2 days. Need to change your delivery?"
        
        elif intent == 'account_management':
            return "🔐 I can help with your account. What specifically do you need help with?"
        
        elif intent == 'technical_support':
            return "🛠️ Technical issues happen! Let me troubleshoot this with you."
        
        elif intent == 'general_complaint':
            return "😔 I understand your frustration. Let me connect you with a human agent who can help."
        
        else:
            return "👋 I'm here to help! Can you tell me more about what you need?"
    
    def _get_progressive_response(self, intent: str, entities: Dict, session_id: str) -> str:
        """Generate follow-up responses when user asks same intent again"""
        
        if intent == 'order_status':
            if 'order_number' in entities and entities['order_number']:
                return "📋 Your order is being processed. Would you like me to send tracking updates to your phone?"
            else:
                return "🔍 Still need that order number to look this up for you."
        
        elif intent == 'returns_refunds':
            return "📤 Ready to start the return? I'll need to collect a few details to generate your return label."
        
        elif intent == 'payment_billing':
            return "💭 Is this about a specific transaction? I can help dispute charges or update payment methods."
        
        elif intent == 'shipping_delivery':
            return "📍 Want to change your delivery address or upgrade to express shipping?"
        
        else:
            return "🤔 Let me get more specific details to help you better."
    
    def _handle_more_details_request(self, session_id: str) -> Dict:
        """Handle requests for more detailed information"""
        session_state = self.get_session_state(session_id)
        
        # Try to use RAG pipeline for detailed response
        if self.rag_pipeline and hasattr(session_state, 'last_retrieved') and session_state.last_retrieved:
            detailed_response = self.rag_pipeline.expand_details(session_state.last_retrieved)
            sources = getattr(session_state, 'last_sources', [])
            
            return {
                'response': detailed_response,
                'intent': 'detailed_information',
                'confidence': 1.0,
                'entities': {},
                'sources': [{'title': source, 'score': 1.0} for source in sources]
            }
        
        # Fallback to original detailed responses
        last_intent = getattr(session_state, 'last_intent', None)
        
        if not last_intent:
            return {
                'response': "I'd be happy to provide more details! What specifically would you like to know more about?",
                'intent': 'more_details_request',
                'confidence': 1.0,
                'entities': {},
                'sources': []
            }
        
        # Provide detailed information based on last intent
        detailed_responses = {
            'order_status': """📦 **Detailed Order Tracking Process:**

• **Step 1:** Order Confirmation - You'll receive email confirmation within 5 minutes
• **Step 2:** Processing - Order is picked and packed (1-2 hours for in-stock items)
• **Step 3:** Shipped - You'll get tracking number via SMS and email
• **Step 4:** In Transit - Track via our website or carrier's website
• **Step 5:** Delivered - Signature confirmation or safe drop-off

**Additional Options:**
• SMS alerts for real-time updates
• Delivery instructions and preferred time slots
• Change delivery address before shipping""",
            
            'returns_refunds': """🔄 **Complete Return & Refund Process:**

**Return Window:** 30 days from delivery date
**Condition Requirements:**
• Items must be unused with original tags
• Original packaging required
• Return authorization needed for some items

**Return Process:**
• **Step 1:** Initiate return request online
• **Step 2:** Print prepaid return label (if eligible)
• **Step 3:** Package item securely
• **Step 4:** Drop off at authorized location
• **Step 5:** Refund processed within 5-7 business days

**Refund Methods:**
• Original payment method (most cases)
• Store credit for certain items
• Gift card replacements""",
            
            'payment_billing': """💳 **Payment Methods & Billing Information:**

**Accepted Payment Methods:**
• Credit Cards: Visa, MasterCard, American Express
• Debit Cards with Visa/MasterCard logo
• Digital Wallets: PayPal, Apple Pay, Google Pay
• Buy Now Pay Later: Klarna, Afterpay
• Gift Cards and Store Credit

**Payment Security:**
• 256-bit SSL encryption
• PCI DSS compliant processing
• Fraud protection monitoring

**Billing Issues:**
• Payment failures: Check card details, expiry, and available balance
• Unauthorized charges: Report within 60 days for investigation
• Payment plan options available for large orders""",
            
            'shipping_delivery': """🚚 **Shipping & Delivery Options:**

**Delivery Speeds:**
• **Standard:** 3-5 business days ($5.99 or free over $50)
• **Express:** 1-2 business days ($12.99)
• **Same Day:** Available in select cities ($19.99)
• **Pickup:** Free at partner locations

**Delivery Features:**
• Safe drop-off with photo confirmation
• Signature required option
• Scheduled delivery windows
• GPS tracking for same-day delivery

**Special Handling:**
• Fragile items packaging
• Age verification for restricted items
• White glove delivery for large items"""
        }
        
        detailed_response = detailed_responses.get(last_intent, 
            "I don't have detailed information about that topic. Could you be more specific about what you'd like to know?")
        
        return {
            'response': detailed_response,
            'intent': 'detailed_information',
            'confidence': 1.0,
            'entities': {},
            'sources': []
        }
    
    def process_query(self, query: str, session_id: str) -> Dict:
        """Process customer query and return response"""
        session_state = self.get_session_state(session_id)
        session_state.interaction_count += 1
        
        # Handle information collection for tickets
        if session_state.collecting_info:
            info_response, collection_complete = self.collect_customer_info(query, session_state)
            if not collection_complete:
                return {
                    'response': info_response,
                    'intent': 'info_collection',
                    'confidence': 1.0,
                    'entities': {},
                    'sources': [],
                    'collecting_info': True
                }
            else:
                # Info collection complete, create ticket
                history = self.conversation_manager.get_conversation_history(session_id, limit=10)
                issue_description = f"Customer issue: {history[-1]['user_query'] if history else 'General inquiry'}"
                
                ticket_id = self.ticket_system.create_ticket(
                    session_id, 
                    session_state.collected_info,
                    issue_description,
                    getattr(session_state, 'last_intent', 'general_inquiry'),
                    history
                )
                
                response = f"Perfect! I've created ticket #{ticket_id} for you. Our specialized support team will contact you at {session_state.collected_info['email']} within 24 hours. Is there anything else I can help you with right now?"
                
                return {
                    'response': response,
                    'intent': 'ticket_created',
                    'confidence': 1.0,
                    'entities': {'ticket_id': ticket_id},
                    'sources': [],
                    'ticket_created': True
                }
        
        # Handle ticket confirmation responses
        if session_state.awaiting_ticket_confirmation:
            if any(word in query.lower() for word in ['yes', 'yes please', 'ok', 'sure', 'okay', 'create']):
                session_state.awaiting_ticket_confirmation = False
                # Create a basic ticket
                history = self.conversation_manager.get_conversation_history(session_id, limit=5)
                ticket_id = self.ticket_system.create_ticket(
                    session_id, 
                    {'name': 'Customer', 'email': 'pending@example.com', 'phone': 'pending'},
                    f"Customer issue: {query}",
                    getattr(session_state, 'last_intent', 'general_inquiry'),
                    history
                )
                return {
                    'response': f"✅ I've created support ticket #{ticket_id} for you. Our specialized team will follow up shortly with next steps.",
                    'intent': 'ticket_created',
                    'confidence': 1.0,
                    'entities': {'ticket_id': ticket_id},
                    'sources': []
                }
            elif any(word in query.lower() for word in ['no', 'not now', 'later', 'cancel']):
                session_state.awaiting_ticket_confirmation = False
                return {
                    'response': "No problem! I'll keep helping you here. What else can I assist with?",
                    'intent': 'ticket_declined',
                    'confidence': 1.0,
                    'entities': {},
                    'sources': []
                }
        
        # Check for quick response patterns first
        quick_type = self.detect_quick_response_type(query)
        if quick_type:
            import random
            response = random.choice(self.quick_responses[quick_type])
            
            # If it's a greeting, also ask what they need help with
            if quick_type == 'greeting':
                response += "\n\n**I can help you with:**\n• Order tracking & status\n• Returns & refunds\n• Payment issues\n• Product information\n• Account help"
            
            return {
                'response': response,
                'intent': quick_type,
                'confidence': 1.0,
                'entities': {},
                'sources': []
            }
        
        # Handle "Talk to human" requests
        if any(phrase in query.lower() for phrase in ['talk to human', 'human support', 'speak to agent', 'human agent']):
            session_state.awaiting_ticket_confirmation = True
            return {
                'response': "I'll connect you with our human support team right away. Would you like me to create a support ticket for you?",
                'intent': 'human_support_request',
                'confidence': 1.0,
                'entities': {},
                'sources': []
            }
        
        # Handle "More details" requests FIRST (before ecommerce check)
        logger.info(f"🔍 Checking if '{query}' is a more details request...")
        is_more_details = any(phrase in query.lower() for phrase in ['more details', 'show more', 'tell me more']) or query.lower().strip() == 'details'
        logger.info(f"🔍 Is more details request: {is_more_details}")
        
        if is_more_details:
            logger.info(f"✅ More details request detected for session: {session_id}")
            
            # Check session dictionary directly for persistence
            if session_id in self.session_states and hasattr(self.session_states[session_id], 'last_retrieved'):
                stored_content = self.session_states[session_id].last_retrieved
                logger.info(f"Found stored content length: {len(stored_content) if stored_content else 0}")
                if stored_content:
                    detailed_response = stored_content[:1000]  # Cap length
                    stored_sources = getattr(self.session_states[session_id], 'last_sources', [])
                    return {
                        'response': f"📚 **Detailed Information:**\n\n{detailed_response}",
                        'intent': 'detailed_information',
                        'confidence': 1.0,
                        'entities': {},
                        'sources': stored_sources
                    }
            
            # Debug: Log what we found
            logger.info(f"No stored content found for session {session_id}")
            logger.info(f"Session states keys: {list(self.session_states.keys())}")
            if session_id in self.session_states:
                logger.info(f"Session exists but no last_retrieved: {hasattr(self.session_states[session_id], 'last_retrieved')}")
            
            # Fallback if no stored content
            return {
                'response': "I don't have previous details to expand on. Please ask a specific question about orders, returns, payments, or shipping.",
                'intent': 'more_details_fallback',
                'confidence': 1.0,
                'entities': {},
                'sources': []
            }
        
        # Check if query is ecommerce-related
        if not self.is_ecommerce_related(query):
            return {
                'response': "I'm specialized in helping with ecommerce and shopping questions. I can assist you with orders, returns, payments, products, shipping, and account issues. How can I help you today?",
                'intent': 'non_ecommerce',
                'confidence': 1.0,
                'entities': {},
                'sources': []
            }
        

        
        # Classify intent and extract entities
        intent, confidence = self.intent_classifier.classify_intent(query)
        entities = self.entity_extractor.extract_entities(query)
        session_state.last_intent = intent
        
        # Search knowledge base
        relevant_docs = self.knowledge_base.search(query, top_k=3)
        
        # Generate contextual response
        response = self.generate_contextual_response(
            query, intent, entities, relevant_docs, session_state, session_id
        )
        
        # Check if we should offer ticket creation
        should_offer_ticket = self.should_create_ticket(session_state, intent)
        
        if should_offer_ticket and not session_state.awaiting_ticket and not session_state.awaiting_ticket_confirmation:
            if intent == 'general_complaint' or 'not working' in query.lower() or 'problem' in query.lower():
                response += "\n\nI understand this is frustrating. Would you like me to connect you with our specialized support team who can provide more detailed assistance?"
                session_state.awaiting_ticket_confirmation = True
            elif session_state.interaction_count >= 3:
                response += "\n\nI want to make sure you get the best help possible. Would you like me to create a support ticket so our team can assist you further?"
                session_state.awaiting_ticket_confirmation = True
        

        
        # Save conversation with sources
        sources_for_analytics = [{'title': doc.title, 'score': score} for doc, score in relevant_docs] if relevant_docs else []
        if hasattr(session_state, 'last_sources') and session_state.last_sources:
            sources_for_analytics = [{'title': source, 'score': 1.0} for source in session_state.last_sources]
        
        self.conversation_manager.save_conversation(
            session_id, query, response, intent, confidence, entities
        )
        
        # Log analytics with sources
        self.conversation_manager._log_analytics(
            session_id, query, intent, confidence, entities, sources_for_analytics
        )
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'sources': [{'title': doc.title, 'score': score} for doc, score in relevant_docs]
        }

# Initialize chatbot
chatbot = CustomerSupportChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        start_time = time.time()
        data = request.json
        query = data.get('query', '').strip()
        session_id = data.get('session_id', 'default_session')
        user_msg = query.lower()
        
        # 🚀 Enhanced chatbot is now active!
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Get or create session state
        if session_id not in chatbot.session_states:
            chatbot.session_states[session_id] = ConversationState()
        session_state = chatbot.session_states[session_id]
        
        # No more "More details" - we'll give comprehensive responses upfront

        # Handle ticket confirmation flow - FIXED with robust session handling
        if (session_id in chatbot.session_states and 
            hasattr(chatbot.session_states[session_id], 'awaiting_ticket_confirmation') and 
            chatbot.session_states[session_id].awaiting_ticket_confirmation):
            
            if any(phrase in user_msg for phrase in ["yes", "please", "sure", "ok", "create"]):
                try:
                    ticket_id = chatbot.ticket_system.create_ticket(
                        session_id=session_id,
                        customer_info={'name': '', 'email': '', 'phone': ''},
                        issue_description="Human support request", 
                        issue_category="general_support",
                        conversation_history=[]
                    )
                    chatbot.session_states[session_id].awaiting_ticket_confirmation = False
                    return jsonify({
                        'response': f"✅ I've created support ticket #{ticket_id} for you. Our specialized team will follow up shortly with next steps.",
                        'intent': 'ticket_created',
                        'confidence': 1.0,
                        'entities': {},
                        'sources': []
                    })
                except Exception as e:
                    logger.error(f"Error creating ticket: {e}")
                    return jsonify({
                        'response': "I encountered an issue creating your ticket. Let me connect you directly with our support team.",
                        'intent': 'ticket_error',
                        'confidence': 1.0,
                        'entities': {},
                        'sources': []
                    })
            
            elif any(phrase in user_msg for phrase in ["no", "not now", "cancel", "later"]):
                chatbot.session_states[session_id].awaiting_ticket_confirmation = False
                return jsonify({
                    'response': "No problem! I'm here if you need any other assistance.",
                    'intent': 'ticket_declined',
                    'confidence': 1.0,
                    'entities': {},
                    'sources': []
                })

        # Handle order ID input with context awareness - FIXED
        if "order id" in user_msg or any(char.isdigit() for char in query):
            order_id = "".join([c for c in query if c.isdigit()])
            if order_id:
                session_state.last_order_id = order_id
                
                # Context-aware response based on previous intent
                last_intent = getattr(session_state, 'last_intent', None)
                
                if last_intent == 'return_request':
                    return jsonify({
                        'response': f"Perfect! I've found order #{order_id}. To process your return:\n\n1️⃣ Go to 'My Orders' section\n2️⃣ Select order #{order_id}\n3️⃣ Click 'Return Items'\n4️⃣ Choose items and reason\n5️⃣ Print the prepaid return label\n\nRefund will be processed within 5-7 business days after we receive your items. Need help with anything else?",
                        'intent': 'return_processing',
                        'confidence': 1.0,
                        'entities': {'order_number': order_id},
                        'sources': []
                    })
                else:
                    # Default to tracking
                    return jsonify({
                        'response': f"Got it! Tracking order #{order_id} 📦\n\n**Status**: In transit\n**Estimated delivery**: 2-3 business days\n**Tracking**: Package is currently en route to your address\n\nYou'll receive SMS/email updates as it progresses. Need more help?",
                        'intent': 'order_tracking',
                        'confidence': 1.0,
                        'entities': {'order_number': order_id},
                        'sources': []
                    })

        # Handle escalation requests
        if any(phrase in user_msg for phrase in ["human", "support", "agent", "talk to human", "speak to agent"]):
            # Store ticket request state in chatbot session dict for reliability
            if session_id not in chatbot.session_states:
                chatbot.session_states[session_id] = ConversationState()
            chatbot.session_states[session_id].awaiting_ticket_confirmation = True
            chatbot.session_states[session_id].last_intent = 'human_support_request'
            
            return jsonify({
                'response': "I'll connect you with our human support team right away. Would you like me to create a support ticket for you?",
                'intent': 'human_support_request',
                'confidence': 1.0,
                'entities': {},
                'sources': []
            })

        # Use enhanced RAG pipeline for truly helpful responses
        logger.info(f"🎯 Processing query: '{query}' with enhanced system")
        
        if hasattr(chatbot, 'rag_pipeline') and chatbot.rag_pipeline:
            logger.info("✅ Using RAG pipeline for helpful response")
            try:
                # Get comprehensive response from RAG
                helpful_response, retrieved_text, sources, latency = chatbot.rag_pipeline.helpful_response(query)
                logger.info(f"🚀 Generated helpful response: {len(helpful_response)} chars")
                
                # Store intent for context awareness
                if "refund" in query.lower():
                    session_state.last_intent = 'return_request'
                elif "return" in query.lower():
                    session_state.last_intent = 'return_request'
                elif "track" in query.lower() or "order" in query.lower():
                    session_state.last_intent = 'order_tracking'
                elif "payment" in query.lower():
                    session_state.last_intent = 'payment_help'
                
                result = {
                    'response': helpful_response,
                    'intent': 'helpful_rag_response',
                    'confidence': 0.9,
                    'entities': {},
                    'sources': sources
                }
                
            except Exception as e:
                logger.error(f"RAG pipeline error in chat route: {e}")
                logger.info("🔄 Falling back to contextual response")
                # Use contextual fallback instead of old process_query
                helpful_response = chatbot.rag_pipeline._generate_contextual_fallback(query)
                result = {
                    'response': helpful_response,
                    'intent': 'contextual_fallback',
                    'confidence': 0.8,
                    'entities': {},
                    'sources': []
                }
        else:
            logger.warning("❌ No RAG pipeline available - using contextual fallback")
            # Always use contextual fallback for better responses
            # Generate basic contextual response
            query_lower = query.lower()
            if "return" in query_lower:
                helpful_response = "I'd be happy to help with your return! To get started, I'll need your order number. You can find it in your email confirmation or account dashboard. Returns are typically processed within 30 days of delivery."
            elif "order" in query_lower:
                helpful_response = "I can help you with your order! Please share your order number (starts with #) and I'll look up the current status, tracking info, and estimated delivery date."
            elif "payment" in query_lower:
                helpful_response = "I can assist with payment questions! Whether it's about payment methods, failed transactions, or refunds, please let me know the specific issue and I'll guide you through the solution."
            elif "shipping" in query_lower:
                helpful_response = "I'm here to help with shipping questions! We offer standard (3-5 days), express (1-2 days), and same-day delivery in select areas. What specific shipping info do you need?"
            else:
                helpful_response = "I'm here to help! I can assist with orders, returns, payments, shipping, and account questions. What specific issue can I help you resolve today?"
            result = {
                'response': helpful_response,
                'intent': 'basic_contextual',
                'confidence': 0.7,
                'entities': {},
                'sources': []
            }
        
        # Add response time to analytics
        response_time = time.time() - start_time
        try:
            with open(ANALYTICS_PATH, 'r') as f:
                analytics_log = json.load(f)
            if analytics_log:
                analytics_log[-1]['response_time'] = response_time
                with open(ANALYTICS_PATH, 'w') as f:
                    json.dump(analytics_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update response time: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback (thumbs up/down)"""
    try:
        data = request.json
        feedback_type = data.get('reaction', '')
        session_id = data.get('session_id', 'default_session')
        
        if feedback_type in ['👍', '👎']:
            # Log feedback to analytics
            feedback_data = {
                "timestamp": time.time(),
                "session_id": session_id,
                "feedback": feedback_type,
                "type": "user_feedback"
            }
            
            analytics_file = ANALYTICS_PATH
            if os.path.exists(analytics_file):
                with open(analytics_file, "r") as f:
                    analytics_log = json.load(f)
            else:
                analytics_log = []
            
            analytics_log.append(feedback_data)
            
            with open(analytics_file, "w") as f:
                json.dump(analytics_log, f, indent=2)
            
            logger.info(f"Feedback received: {feedback_type} from session {session_id}")
            return jsonify({"status": "ok", "message": "Feedback recorded"})
        else:
            return jsonify({"status": "error", "message": "Invalid feedback type"}), 400
            
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analytics', methods=['GET'])
def analytics():
    """Get analytics data for dashboard"""
    try:
        analytics_file = ANALYTICS_PATH
        if not os.path.exists(analytics_file):
            return jsonify({
                "total_conversations": 0,
                "intent_distribution": {},
                "avg_response_time": 0,
                "satisfaction_rate": 0,
                "recent_activity": []
            })
        
        with open(analytics_file, "r") as f:
            analytics_log = json.load(f)
        
        # Filter out feedback entries for main analytics
        conversations = [entry for entry in analytics_log if entry.get('type') != 'user_feedback']
        feedback_entries = [entry for entry in analytics_log if entry.get('type') == 'user_feedback']
        
        # Calculate analytics
        total_conversations = len(conversations)
        
        # Intent distribution
        intent_counts = {}
        response_times = []
        
        for entry in conversations:
            intent = entry.get('intent', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            if 'response_time' in entry and isinstance(entry['response_time'], (int, float)):
                response_times.append(entry['response_time'])
        
        # Average response time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Satisfaction rate
        positive_feedback = len([f for f in feedback_entries if f.get('feedback') == '👍'])
        total_feedback = len(feedback_entries)
        satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        # Recent activity (last 10 conversations)
        recent_activity = conversations[-10:] if conversations else []
        
        return jsonify({
            "total_conversations": total_conversations,
            "intent_distribution": intent_counts,
            "avg_response_time": round(avg_response_time, 3),
            "satisfaction_rate": round(satisfaction_rate, 1),
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "recent_activity": recent_activity
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)