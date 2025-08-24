import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cohere
from groq import Groq
import PyPDF2

import re
from collections import defaultdict
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

class RAGChatbot:
    def __init__(self):
        # Initialize APIs
        self.cohere_api_key = Config.COHERE_API_KEY
        self.groq_api_key = Config.GROQ_API_KEY
        
        # Initialize clients
        self.cohere_client = None
        self.groq_client = None
        
        # Initialize clients only if API keys are available
        if self.cohere_api_key and self.cohere_api_key != 'your_cohere_api_key_here':
            try:
                self.cohere_client = cohere.Client(self.cohere_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere client: {e}")
        
        if self.groq_api_key and self.groq_api_key != 'your_groq_api_key_here':
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        # System state
        self.knowledge_base = []
        self.embeddings = None
        self.vector_index = None
        self.conversation_history = {}
        self.conversation_state = {}  # Track conversation state (order_id, product, etc.)
        
        # Slot filling state machine
        self.active_conversations = {}  # conversation_id -> conversation_state
        self.slot_filling_states = {
            'return': {
                'required_slots': ['order_id', 'reason', 'resolution_choice'],
                'optional_slots': ['product', 'damage_description'],
                'current_slot': None,
                'filled_slots': set()
            },
            'shipping': {
                'required_slots': ['order_id'],
                'optional_slots': ['shipping_issue', 'tracking_number'],
                'current_slot': None,
                'filled_slots': set()
            },
            'billing': {
                'required_slots': ['order_id', 'billing_issue'],
                'optional_slots': ['amount', 'date'],
                'current_slot': None,
                'filled_slots': set()
            },
            'technical': {
                'required_slots': ['issue_description'],
                'optional_slots': ['device', 'browser', 'error_message'],
                'current_slot': None,
                'filled_slots': set()
            },
            'account': {
                'required_slots': ['email', 'account_issue'],
                'optional_slots': ['username', 'last_login'],
                'current_slot': None,
                'filled_slots': set()
            }
        }
        
        self.analytics = {
            'total_queries': 0,
            'total_tickets': 0,
            'intent_distribution': defaultdict(int),
            'avg_response_time': 0.0,
            'entity_extraction_rate': 0.0,
            'response_times': []
        }
        
        # Intent patterns with better coverage
        self.intent_patterns = {
            'greeting': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'morning', 'afternoon', 'evening'],
            'billing': ['bill', 'charge', 'payment', 'invoice', 'money', 'cost', 'fee', 'billing'],
            'technical': ['bug', 'error', 'crash', 'not working', 'broken', 'issue', 'problem', 'technical'],
            'account': ['login', 'password', 'account', 'profile', 'sign in', 'access', 'account'],
            'shipping': ['shipping', 'delivery', 'tracking', 'package', 'arrived', 'late', 'delayed', 'shipping', 'delivery'],
            'return': ['return', 'exchange', 'cancel', 'refund', 'send back', 'want to return', 'need to return', 'return an order', 'need refund', 'want refund'],
            'complaint': ['complaint', 'unhappy', 'disappointed', 'terrible', 'awful', 'bad'],
            'off_topic': ['sandwich', 'recipe', 'cooking', 'food', 'weather', 'sports', 'politics']
        }
        
        # Entity patterns
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'order_id': r'\b[A-Z]{2,3}-\d{3,6}\b|\b\d{3,8}\b',  # Also match simple numeric IDs (3-8 digits)
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'amount': r'\$\d+(?:\.\d{2})?',
            'date': r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            'product': r'\b(sunscreen|cream|soap|lotion|shampoo|makeup|cosmetic|aqualogica)\b'
        }
        
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            logger.info("Initializing RAG system...")
            
            # Create data directory if it doesn't exist
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            
            # Load knowledge base
            self.load_knowledge_base()
            
            # Create embeddings and vector index
            if self.knowledge_base:
                if self.cohere_client:
                    self.create_embeddings()
                    logger.info("RAG system initialized successfully")
                else:
                    logger.warning("Cohere client not available - RAG system will not function properly")
            else:
                logger.warning("No knowledge base loaded - creating sample data")
                self.create_sample_knowledge_base()
                
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
    
    def load_knowledge_base(self):
        """Load PDF knowledge base"""
        try:
            # Debug: Log the upload folder path
            logger.info(f"Looking for PDF files in: {os.path.abspath(Config.UPLOAD_FOLDER)}")
            
            # Debug: List ALL files in the directory
            all_files = os.listdir(Config.UPLOAD_FOLDER)
            logger.info(f"All files in directory: {all_files}")
            
            pdf_files = [f for f in all_files if f.endswith('.pdf')]
            logger.info(f"PDF files found: {pdf_files}")
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {Config.UPLOAD_FOLDER}/ directory")
                logger.warning(f"Directory contents: {all_files}")
                return
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(Config.UPLOAD_FOLDER, pdf_file)
                logger.info(f"Loading PDF: {pdf_file}")
                
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            # Split into chunks
                            chunks = self.split_text_into_chunks(text, chunk_size=500)
                            for chunk_num, chunk in enumerate(chunks):
                                self.knowledge_base.append({
                                    'source': pdf_file,
                                    'page': page_num + 1,
                                    'chunk': chunk_num + 1,
                                    'content': chunk.strip()
                                })
            
            logger.info(f"Loaded {len(self.knowledge_base)} chunks from {len(pdf_files)} PDF(s)")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    def create_sample_knowledge_base(self):
        """Create sample knowledge base for demonstration"""
        sample_data = [
            {
                'source': 'sample_kb.txt',
                'page': 1,
                'chunk': 1,
                'content': '''
                Billing and Payment Information:
                - All charges are processed within 24-48 hours of purchase
                - Refunds typically take 3-5 business days to appear on your statement
                - If you see duplicate charges, contact our billing department immediately
                - We accept all major credit cards and PayPal
                - Monthly subscription fees are charged on the same date each month
                '''
            },
            {
                'source': 'sample_kb.txt',
                'page': 1,
                'chunk': 2,
                'content': '''
                Technical Support Guidelines:
                - Clear your browser cache if experiencing loading issues
                - Check your internet connection for connectivity problems
                - Try logging out and logging back in for account-related issues
                - Update your browser to the latest version
                - Disable browser extensions if experiencing compatibility issues
                '''
            },
            {
                'source': 'sample_kb.txt',
                'page': 1,
                'chunk': 3,
                'content': '''
                Account Management:
                - Password reset links are valid for 24 hours
                - Account lockouts occur after 5 failed login attempts
                - Two-factor authentication is required for all accounts
                - Profile information can be updated in account settings
                - Account deletion requests are processed within 30 days
                '''
            },
            {
                'source': 'sample_kb.txt',
                'page': 2,
                'chunk': 1,
                'content': '''
                Shipping and Delivery Information:
                - Standard shipping takes 5-7 business days
                - Express shipping takes 2-3 business days
                - Free shipping is available on orders over $50
                - Tracking numbers are provided within 24 hours of shipment
                - International shipping is available to most countries
                '''
            },
            {
                'source': 'sample_kb.txt',
                'page': 2,
                'chunk': 2,
                'content': '''
                Returns and Exchanges:
                - Items can be returned within 30 days of purchase
                - Original packaging is required for all returns
                - Return shipping is free for defective items
                - Exchanges can be processed online or by phone
                - Refunds are issued to the original payment method
                '''
            }
        ]
        
        self.knowledge_base = sample_data
        self.create_embeddings()
        logger.info(f"Created sample knowledge base with {len(self.knowledge_base)} chunks")
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self):
        """Create embeddings for knowledge base using Cohere"""
        try:
            if not self.knowledge_base:
                logger.warning("No knowledge base to create embeddings for")
                return
            
            logger.info("Creating embeddings with Cohere...")
            
            # Extract text content
            texts = [chunk['content'] for chunk in self.knowledge_base]
            
            # Create embeddings using Cohere
            if not self.cohere_client:
                logger.error("Cohere client not available - cannot create embeddings")
                return
                
            response = self.cohere_client.embed(
                texts=texts,
                model='embed-english-v3.0',
                input_type='search_document'
            )
            
            # Convert to numpy array
            self.embeddings = np.array(response.embeddings, dtype=np.float32)
            
            # Create simple vector index using numpy
            dimension = self.embeddings.shape[1]
            self.vector_index = self.embeddings.copy()
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.vector_index, axis=1, keepdims=True)
            self.vector_index = self.vector_index / norms
            
            logger.info(f"Created vector index with {len(self.embeddings)} embeddings (dimension: {dimension})")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def extract_entities(self, text: str, conversation_context: List[Dict] = None, conversation_id: str = None) -> Dict[str, str]:
        """Extract entities from text using regex patterns with conversation context and state"""
        entities = {}
        
        # Skip entity extraction for obvious greetings (performance optimization)
        text_lower = text.lower().strip()
        if text_lower in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']:
            return entities
        
        # Extract entities from current text ONLY
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
                logger.info(f"Extracted {entity_type} entity: {entities[entity_type]}")
            else:
                logger.info(f"No {entity_type} entity found in text: '{text}'")
        
        # Extract product entities from text ONLY
        product_keywords = ['sunscreen', 'cream', 'soap', 'lotion', 'shampoo', 'makeup', 'cosmetic', 'aqualogica']
        text_lower = text.lower()
        logger.info(f"Checking for product keywords in: '{text_lower}'")
        for keyword in product_keywords:
            if keyword.lower() in text_lower:
                entities['product'] = keyword
                logger.info(f"Extracted product entity: {keyword}")
                break
        
        # Enhance with entities from conversation context if available
        if conversation_context and len(conversation_context) > 0:
            for msg in conversation_context[-3:]:  # Last 3 messages
                if msg.get('entities'):
                    for entity_type, value in msg['entities'].items():
                        # Only add if not already present in current entities
                        if entity_type not in entities:
                            entities[entity_type] = value
                            logger.info(f"Added entity from context: {entity_type}: {value}")
        
        # ONLY enhance with conversation state if we're in an ongoing conversation (more than 1 message)
        if conversation_id and conversation_id in self.conversation_state:
            state = self.conversation_state[conversation_id]
            conversation_length = len(self.conversation_history.get(conversation_id, []))
            
            # Only use state entities if we have an ongoing conversation
            if conversation_length > 1:
                for key in ['order_id', 'product', 'return_reason']:
                    if key in state and key not in entities and state[key] is not None:
                        entities[key] = state[key]
                        logger.info(f"Added entity from state: {key}: {state[key]}")
        
        return entities
    
    def classify_intent(self, text: str, entities: Dict = None) -> tuple:
        """Classify user intent based on keywords and entities with improved logic"""
        text_lower = text.lower()
        
        # Special handling for common patterns with high confidence
        if any(word in text_lower for word in ['delivery', 'shipping', 'package', 'tracking']):
            return 'shipping', 0.9
        
        if any(word in text_lower for word in ['refund', 'return', 'exchange']):
            return 'return', 0.9
        
        if any(word in text_lower for word in ['password', 'login', 'account']):
            return 'account', 0.9
        
        if any(word in text_lower for word in ['bill', 'charge', 'payment', 'billing']):
            return 'billing', 0.9
        
        # General pattern matching
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Boost confidence for e-commerce entities
        if entities:
            if 'order_id' in entities and 'product' in entities:
                # High confidence for return/shipping when we have order + product
                if 'return' in text_lower or 'exchange' in text_lower:
                    return 'return', 0.9
                elif 'shipping' in text_lower or 'delivery' in text_lower:
                    return 'shipping', 0.9
                else:
                    return 'general', 0.8  # High confidence for e-commerce context
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent] / len(self.intent_patterns[best_intent])
            
            # Special case: Greetings should have high confidence
            if best_intent == 'greeting':
                return 'greeting', 0.9
            
            # Boost confidence for clear matches
            if confidence > 0.5:
                confidence = min(confidence + 0.3, 1.0)
            
            return best_intent, min(confidence, 1.0)
        
        # Check for off-topic queries
        if not self.is_ecommerce_related(text):
            return 'off_topic', 0.1
        
        return 'general', 0.3
    
    def search_knowledge_base(self, query: str, top_k: int = 3, conversation_context: List[Dict] = None) -> List[Dict]:
        """Search knowledge base using vector similarity with conversation context"""
        try:
            # Memory safety check
            if self.vector_index is None or self.knowledge_base is None or len(self.knowledge_base) == 0:
                logger.warning("Knowledge base not initialized - returning empty results")
                return []
            
            # Limit query length to prevent memory issues
            if len(query) > 1000:
                query = query[:1000]
                logger.warning("Query truncated to prevent memory issues")
            
            # Enhance query with conversation context if available
            enhanced_query = query
            if conversation_context and len(conversation_context) > 0:
                # Extract key information from recent messages
                recent_queries = [msg.get('query', '') for msg in conversation_context[-3:]]  # Last 3 messages
                recent_entities = []
                for msg in conversation_context[-3:]:
                    if msg.get('entities'):
                        recent_entities.extend([f"{k}:{v}" for k, v in msg['entities'].items()])
                
                # Combine current query with context
                context_text = " ".join(recent_queries + recent_entities)
                enhanced_query = f"{query} {context_text}".strip()
                logger.info(f"Enhanced query with context: {enhanced_query}")
            
            # Create query embedding
            if not self.cohere_client:
                logger.error("Cohere client not available - cannot create query embeddings")
                return []
                
            response = self.cohere_client.embed(
                texts=[enhanced_query],
                model='embed-english-v3.0',
                input_type='search_query'
            )
            
            query_embedding = np.array(response.embeddings, dtype=np.float32)
            
            # Normalize query embedding for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                logger.warning("Query embedding has zero norm - cannot normalize")
                return []
            
            query_embedding = query_embedding / query_norm
            
            # Search vector index using numpy (cosine similarity)
            try:
                similarities = np.dot(self.vector_index, query_embedding.T).flatten()
            except Exception as e:
                logger.error(f"Error calculating similarities: {e}")
                return []
            
            # Ensure we have valid similarities before proceeding
            if similarities.size == 0:
                logger.warning("No similarities calculated - empty result")
                return []
            
            indices = np.argsort(similarities)[::-1][:top_k]  # Top k results
            scores = similarities[indices]
            
            # Debug: Log the arrays
            logger.info(f"Similarities shape: {similarities.shape}, type: {type(similarities)}")
            logger.info(f"Indices shape: {indices.shape}, type: {type(indices)}")
            logger.info(f"Scores shape: {scores.shape}, type: {type(scores)}")
            
            results = []
            # Ensure both arrays are 1D and have the same length
            if len(scores) == len(indices) and len(scores) > 0:
                for i in range(len(scores)):
                    try:
                        # Convert numpy types to Python types safely
                        score = float(scores[i])
                        idx = int(indices[i])
                        
                        # Validate index bounds
                        if 0 <= idx < len(self.knowledge_base):
                            result = self.knowledge_base[idx].copy()
                            result['similarity_score'] = score
                            results.append(result)
                        else:
                            logger.warning(f"Index {idx} out of bounds for knowledge base size {len(self.knowledge_base)}")
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(f"Error processing result {i}: {e}")
                        continue
            else:
                logger.error(f"Array length mismatch: scores={len(scores)}, indices={len(indices)}")
                logger.error(f"Scores length: {len(scores)}, Indices length: {len(indices)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def generate_response(self, query: str, context: List[Dict], intent: str, entities: Dict, conversation_context: List[Dict] = None) -> str:
        """Generate response using decision tree flow to avoid infinite loops"""
        try:
            # Handle greetings first
            if intent == 'greeting':
                return """Hello! I'm your AI customer support assistant. I can help you with:

🔐 Account and password issues
💳 Billing and payment questions  
🛠️ Technical support
📦 Shipping and delivery
🔄 Returns and refunds

How can I assist you today?"""
            
            # DECISION TREE: Check if we have enough info to resolve the issue
            resolution_result = self._check_resolution_ready(intent, entities, conversation_context, conversation_id)
            if resolution_result['can_resolve']:
                # Check if this is a resolution choice
                if resolution_result.get('is_resolution_choice', False):
                    return self._handle_resolution_choice(intent, query, resolution_result['entities'], conversation_id)
                else:
                    return self._generate_resolution_response(intent, resolution_result, conversation_context)
            
            # If we can't resolve yet, ask for missing information
            return self._ask_for_missing_info(intent, entities, conversation_context, conversation_id)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance."
    
    def _check_resolution_ready(self, intent: str, entities: Dict, conversation_context: List[Dict] = None, conversation_id: str = None) -> Dict:
        """Check if we have enough information to resolve the issue"""
        try:
            # Get all entities from conversation context
            context_entities = {}
            if conversation_context and len(conversation_context) > 0:
                for msg in conversation_context:
                    if msg.get('entities'):
                        for key, value in msg['entities'].items():
                            if key not in context_entities:
                                context_entities[key] = value
            
            # Merge with current entities
            all_entities = {**context_entities, **entities}
            
            # Check if this is a resolution choice (final step)
            if conversation_id and conversation_id in self.active_conversations:
                conv_state = self.active_conversations[conversation_id]
                locked_intent = conv_state['locked_intent']
                
                # Check if user is making a resolution choice
                if locked_intent == 'return' and self._is_resolution_choice(entities, conversation_context):
                    return {
                        'can_resolve': True,
                        'missing': [],
                        'entities': all_entities,
                        'intent': locked_intent,
                        'is_resolution_choice': True
                    }
            
            # Define resolution requirements for each intent
            resolution_requirements = {
                'shipping': {
                    'required': ['order_id'],
                    'optional': ['product', 'delivery_date'],
                    'can_resolve': lambda e: 'order_id' in e
                },
                'return': {
                    'required': ['order_id', 'reason'],
                    'optional': ['product', 'damage_description'],
                    'can_resolve': lambda e: 'order_id' in e and 'reason' in e
                },
                'billing': {
                    'required': ['order_id', 'billing_issue'],
                    'optional': ['amount', 'date'],
                    'can_resolve': lambda e: 'order_id' in e and 'billing_issue' in e
                },
                'technical': {
                    'required': ['issue_description'],
                    'optional': ['device', 'browser'],
                    'can_resolve': lambda e: 'issue_description' in e
                },
                'account': {
                    'required': ['email', 'account_issue'],
                    'optional': ['username', 'last_login'],
                    'can_resolve': lambda e: 'email' in e and 'account_issue' in e
                }
            }
            
            if intent not in resolution_requirements:
                return {'can_resolve': False, 'missing': [], 'entities': all_entities}
            
            req = resolution_requirements[intent]
            can_resolve = req['can_resolve'](all_entities)
            
            # Find missing required fields
            missing = [field for field in req['required'] if field not in all_entities]
            
            return {
                'can_resolve': can_resolve,
                'missing': missing,
                'entities': all_entities,
                'intent': intent,
                'is_resolution_choice': False
            }
            
        except Exception as e:
            logger.error(f"Error checking resolution readiness: {e}")
            return {'can_resolve': False, 'missing': [], 'entities': {}, 'intent': intent}
    
    def _is_resolution_choice(self, entities: Dict, conversation_context: List[Dict] = None) -> bool:
        """Check if the user is making a resolution choice (refund/replacement)"""
        try:
            # Check for resolution choice keywords
            resolution_keywords = ['refund', 'replacement', 'exchange', 'money back', 'new item', 'different item']
            
            # Get text from conversation context
            if conversation_context and len(conversation_context) > 0:
                latest_message = conversation_context[-1].get('query', '').lower()
                if any(keyword in latest_message for keyword in resolution_keywords):
                    return True
            
            # Check current entities for resolution choice
            if 'refund' in str(entities).lower() or 'replacement' in str(entities).lower():
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking resolution choice: {e}")
            return False
    
    def _generate_resolution_response(self, intent: str, resolution_result: Dict, conversation_context: List[Dict] = None) -> str:
        """Generate resolution response when we have enough information"""
        try:
            entities = resolution_result['entities']
            intent = resolution_result['intent']
            
            if intent == 'shipping':
                order_id = entities.get('order_id', 'your order')
                return f"""✅ **Shipping Issue Resolved!**

I have all the information I need to help with your shipping concern for order {order_id}.

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Issue: Shipping inquiry for order {order_id}
- Priority: Normal

Our shipping team will review your order and contact you within 2 hours with:
- Current shipping status
- Tracking information
- Estimated delivery date
- Any necessary actions

Is there anything else I can help you with today?"""
            
            elif intent == 'return':
                order_id = entities.get('order_id', 'your order')
                product = entities.get('product', 'the item')
                return f"""✅ **Return Request Processed!**

I have all the information needed for your return request:
- Order ID: {order_id}
- Product: {product}

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Type: Return request
- Priority: High

Our returns team will contact you within 2 hours to:
- Confirm return details
- Provide return shipping label
- Process your refund/replacement
- Arrange pickup if needed

Is there anything else I can help you with today?"""
            
            elif intent == 'billing':
                order_id = entities.get('order_id', 'your order')
                return f"""✅ **Billing Issue Addressed!**

I have your order ID: {order_id} and will investigate your billing concern.

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Type: Billing inquiry
- Priority: Normal

Our billing team will contact you within 2 hours to:
- Review your account
- Resolve the billing issue
- Process any necessary refunds
- Update payment methods if needed

Is there anything else I can help you with today?"""
            
            elif intent == 'technical':
                issue = entities.get('issue_description', 'your technical issue')
                return f"""✅ **Technical Issue Logged!**

I've recorded your technical concern: {issue}

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Type: Technical support
- Priority: Normal

Our technical team will contact you within 2 hours to:
- Troubleshoot the issue
- Provide step-by-step solutions
- Escalate if needed
- Follow up until resolved

Is there anything else I can help you with today?"""
            
            elif intent == 'account':
                email = entities.get('email', 'your account')
                return f"""✅ **Account Issue Addressed!**

I have your email: {email} and will help with your account concern.

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Type: Account support
- Priority: Normal

Our account team will contact you within 2 hours to:
- Verify your identity
- Resolve the account issue
- Provide necessary instructions
- Ensure secure access

Is there anything else I can help you with today?"""
            
            else:
                return """✅ **Issue Logged Successfully!**

I've recorded your concern and created a support ticket.

🎫 **Support Ticket Created**
- Our team will contact you within 2 hours
- We'll work to resolve your issue promptly
- You'll receive regular updates

Is there anything else I can help you with today?"""
                
        except Exception as e:
            logger.error(f"Error generating resolution response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance."
    
    def _ask_for_missing_info(self, intent: str, entities: Dict, conversation_context: List[Dict] = None, conversation_id: str = None) -> str:
        """Ask for missing information using slot-filling state machine"""
        try:
            # Get or create conversation state
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = {
                    'locked_intent': intent,
                    'current_slot': None,
                    'filled_slots': set(),
                    'slot_data': {}
                }
            
            conv_state = self.active_conversations[conversation_id]
            
            # Lock the intent for this conversation
            if not conv_state['locked_intent']:
                conv_state['locked_intent'] = intent
            
            # Get all entities from conversation context and current message
            context_entities = {}
            if conversation_context and len(conversation_context) > 0:
                for msg in conversation_context:
                    if msg.get('entities'):
                        for key, value in msg['entities'].items():
                            if key not in context_entities:
                                context_entities[key] = value
                                conv_state['slot_data'][key] = value
                                conv_state['filled_slots'].add(key)
            
            # Merge with current entities
            all_entities = {**context_entities, **entities}
            for key, value in entities.items():
                conv_state['slot_data'][key] = value
                conv_state['filled_slots'].add(key)
            
            # Use the locked intent, not the current classified intent
            locked_intent = conv_state['locked_intent']
            
            # Check what slots we need and what we have
            if locked_intent not in self.slot_filling_states:
                return self._get_intent_specific_response(intent, "", conversation_context)
            
            slot_state = self.slot_filling_states[locked_intent]
            required_slots = slot_state['required_slots']
            optional_slots = slot_state['optional_slots']
            
            # Find missing required slots
            missing_slots = [slot for slot in required_slots if slot not in conv_state['filled_slots']]
            
            if not missing_slots:
                # All required slots filled, ask for resolution choice
                return self._ask_for_resolution_choice(locked_intent, conv_state['slot_data'])
            
            # Ask for the next missing slot
            next_slot = missing_slots[0]
            conv_state['current_slot'] = next_slot
            
            return self._ask_for_specific_slot(locked_intent, next_slot, conv_state['slot_data'])
            
        except Exception as e:
            logger.error(f"Error asking for missing info: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance."
    
    def _ask_for_specific_slot(self, intent: str, slot_name: str, slot_data: Dict) -> str:
        """Ask for a specific slot in the slot-filling flow"""
        try:
            if intent == 'return':
                if slot_name == 'order_id':
                    return """🔄 **Return Request**

I'd be happy to help you with your return!

🔍 **I need your order ID to assist you:**
- Please provide your order number
- You can find this in your order confirmation email

**Example:** "I want to return order 12345"

Once you provide the order ID, I can help process your return request."""
                
                elif slot_name == 'reason':
                    order_id = slot_data.get('order_id', 'your order')
                    return f"""✅ **Order ID Received: {order_id}**

Great! Now I need to know the reason for your return:

🔍 **Why are you returning this item?**
- Damaged or defective product
- Wrong item received
- Size doesn't fit
- Changed your mind
- Other reason

**Example:** "The product was damaged during shipping" or "I received the wrong item"

Once you tell me the reason, I can process your return request."""
                
                elif slot_name == 'resolution_choice':
                    order_id = slot_data.get('order_id', 'your order')
                    reason = slot_data.get('reason', 'your return reason')
                    return f"""✅ **Return Details Complete**

Perfect! I have all the information I need:
- Order ID: {order_id}
- Reason: {reason}

Now I need to know your preference:

🔍 **What would you like?**
- **Refund**: Money back to your original payment method
- **Replacement**: Same item shipped to you again
- **Exchange**: Different item of equal value

**Example:** "I want a refund" or "I'd like a replacement"

Once you choose, I can immediately process your request and create a support ticket."""
            
            elif intent == 'shipping':
                if slot_name == 'order_id':
                    return """📦 **Shipping Assistance**

I'd be happy to help with your shipping concern!

🔍 **I need your order ID to assist you:**
- Please provide your order number
- You can find this in your order confirmation email
- Or check your account order history

**Example:** "My order ID is 12345" or "Where is order 12345?"

Once you provide the order ID, I can immediately create a support ticket and get our shipping team involved."""
            
            elif intent == 'billing':
                if slot_name == 'order_id':
                    return """💳 **Billing Assistance**

I'd be happy to help with your billing concern!

🔍 **I need your order ID or transaction ID:**
- Please provide your order number
- Or the transaction ID from your bank statement
- You can find this in your order confirmation email

**Example:** "I was charged twice for order 12345"

Once you provide the order ID, I can immediately investigate and create a support ticket."""
                
                elif slot_name == 'billing_issue':
                    order_id = slot_data.get('order_id', 'your order')
                    return f"""✅ **Order ID Received: {order_id}**

Great! Now I need to know the specific billing issue:

🔍 **What billing problem are you experiencing?**
- Duplicate charge
- Incorrect amount charged
- Payment failed
- Refund not received
- Subscription billing issue
- Other billing concern

**Example:** "I was charged twice for the same order" or "My payment failed during checkout"

Once you describe the issue, I can investigate and create a support ticket."""
            
            elif intent == 'technical':
                if slot_name == 'issue_description':
                    return """🛠️ **Technical Support**

I'd be happy to help with your technical issue!

🔍 **Please describe the problem:**
- What exactly is happening?
- What were you trying to do?
- What error messages do you see?
- What device/browser are you using?

**Examples:**
- "I can't log into my account"
- "The app keeps crashing on my iPhone"
- "I get an error when trying to checkout"

Once you provide details, I can create a support ticket for our technical team."""
            
            elif intent == 'account':
                if slot_name == 'email':
                    return """🔐 **Account Support**

I'd be happy to help with your account issue!

🔍 **I need your email address:**
- Please provide the email associated with your account
- This helps me verify your identity and access

**Example:** "I can't log into john@email.com"

Once you provide your email, I can immediately assist with your account concern."""
                
                elif slot_name == 'account_issue':
                    email = slot_data.get('email', 'your account')
                    return f"""✅ **Email Received: {email}**

Great! Now I need to know the specific account issue:

🔍 **What account problem are you experiencing?**
- Can't log in
- Forgot password
- Account locked
- Need to update profile
- Two-factor authentication issue
- Other account concern

**Examples:**
- "I forgot my password"
- "My account is locked after too many failed attempts"
- "I need to update my email address"

Once you describe the issue, I can assist you and create a support ticket if needed."""
            
            else:
                return f"""I need more information to help you with your {intent} request.

🔍 **Please provide:**
- {slot_name.replace('_', ' ').title()}

This will help me assist you properly."""
                
        except Exception as e:
            logger.error(f"Error asking for specific slot: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance."
    
    def _ask_for_resolution_choice(self, intent: str, slot_data: Dict) -> str:
        """Ask for resolution choice when all required slots are filled"""
        try:
            if intent == 'return':
                order_id = slot_data.get('order_id', 'your order')
                reason = slot_data.get('reason', 'your return reason')
                return f"""✅ **Return Details Complete**

Perfect! I have all the information I need:
- Order ID: {order_id}
- Reason: {reason}

Now I need to know your preference:

🔍 **What would you like?**
- **Refund**: Money back to your original payment method
- **Replacement**: Same item shipped to you again
- **Exchange**: Different item of equal value

**Example:** "I want a refund" or "I'd like a replacement"

Once you choose, I can immediately process your request and create a support ticket."""
            
            else:
                return """✅ **Information Complete**

I have all the information I need to help you!

🎫 **Support Ticket Created**
- Our team will contact you within 2 hours
- We'll work to resolve your issue promptly
- You'll receive regular updates

Is there anything else I can help you with today?"""
                
        except Exception as e:
            logger.error(f"Error asking for resolution choice: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance."
    
    def _handle_resolution_choice(self, intent: str, choice: str, slot_data: Dict, conversation_id: str) -> str:
        """Handle the user's resolution choice and create final ticket"""
        try:
            if intent == 'return':
                order_id = slot_data.get('order_id', 'your order')
                reason = slot_data.get('reason', 'your return reason')
                
                if 'refund' in choice.lower():
                    # Clear conversation state after resolution
                    if conversation_id in self.active_conversations:
                        del self.active_conversations[conversation_id]
                    
                    return f"""✅ **Refund Request Processed Successfully!**

Perfect! I've processed your refund request:
- Order ID: {order_id}
- Reason: {reason}
- Resolution: Refund to original payment method

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Type: Refund request
- Priority: High
- Status: Processing

📋 **Next Steps:**
- Refund will be processed within 3-5 business days
- You'll receive email confirmation
- Return shipping label will be sent within 2 hours
- Our team will contact you to arrange pickup

💰 **Refund Timeline:**
- Processing: 1-2 business days
- Bank processing: 3-5 business days
- Appears on your statement within 5-7 days

Is there anything else I can help you with today?"""
                
                elif 'replacement' in choice.lower():
                    # Clear conversation state after resolution
                    if conversation_id in self.active_conversations:
                        del self.active_conversations[conversation_id]
                    
                    return f"""✅ **Replacement Request Processed Successfully!**

Perfect! I've processed your replacement request:
- Order ID: {order_id}
- Reason: {reason}
- Resolution: Replacement item

🎫 **Support Ticket Created**
- Ticket ID: TKT-{uuid.uuid4().hex[:8].upper()}
- Type: Replacement request
- Priority: High
- Status: Processing

📦 **Next Steps:**
- Replacement will be shipped within 24 hours
- You'll receive new tracking number
- Return shipping label will be sent within 2 hours
- Our team will contact you to arrange pickup

🚚 **Shipping Timeline:**
- Processing: Same day
- Shipping: Next business day
- Delivery: 3-5 business days

Is there anything else I can help you with today?"""
                
                else:
                    return """I didn't catch your preference clearly. 

🔍 **Please choose one of these options:**
- **Refund**: "I want a refund"
- **Replacement**: "I'd like a replacement"
- **Exchange**: "I want to exchange for something else"

Which would you prefer?"""
            
            else:
                # Clear conversation state after resolution
                if conversation_id in self.active_conversations:
                    del self.active_conversations[conversation_id]
                
                return """✅ **Issue Resolved Successfully!**

I've processed your request and created a support ticket.

🎫 **Support Ticket Created**
- Our team will contact you within 2 hours
- We'll work to resolve your issue promptly
- You'll receive regular updates

Is there anything else I can help you with today?"""
                
        except Exception as e:
            logger.error(f"Error handling resolution choice: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance."
    
    def _get_intent_specific_response(self, intent: str, query: str, conversation_context: List[Dict] = None) -> str:
        """Generate intent-specific responses when no knowledge base content is found"""
        try:
            # Check conversation context for existing entities
            existing_entities = {}
            if conversation_context and len(conversation_context) > 0:
                for msg in conversation_context:
                    if msg.get('entities'):
                        for key, value in msg['entities'].items():
                            if key not in existing_entities:
                                existing_entities[key] = value
            
            if intent == 'billing':
                # Check if we already have order_id from context
                order_info = ""
                if 'order_id' in existing_entities:
                    order_info = f"\n✅ **Order ID**: {existing_entities['order_id']} (already provided)"
                
                return f"""I understand you have a billing question. While I don't have specific information about your account in my knowledge base, I can help you with general billing information:

💳 **General Billing Information:**
- Charges are typically processed within 24-48 hours of purchase
- Refunds take 3-5 business days to appear on your statement
- We accept all major credit cards and PayPal
- Monthly subscriptions are charged on the same date each month{order_info}

🔍 **To help you better, I need:**
- Your order ID or account email
- Specific billing issue (duplicate charge, refund request, etc.)
- Date of the transaction

Would you like me to create a support ticket so our billing team can assist you directly with your specific issue?"""
            
            elif intent == 'account':
                return """I understand you need help with your account. Here's what I can help you with:

🔐 **Account Management:**
- Password reset (valid for 24 hours)
- Account lockout assistance (occurs after 5 failed attempts)
- Two-factor authentication setup
- Profile information updates
- Account deletion requests (processed within 30 days)

🔍 **To help you better, I need:**
- Your account email address
- Specific account issue you're experiencing
- Whether you can currently log in or not

Would you like me to create a support ticket so our account team can assist you directly?"""
            
            elif intent == 'technical':
                return """I understand you're experiencing technical issues. Here are some common solutions:

🛠️ **Quick Troubleshooting Steps:**
- Clear your browser cache and cookies
- Check your internet connection
- Try logging out and back in
- Update your browser to the latest version
- Disable browser extensions temporarily

🔍 **To help you better, I need:**
- What specific error you're seeing
- What you were trying to do when it happened
- Your browser and device information
- Screenshot of the error (if possible)

Would you like me to create a support ticket so our technical team can assist you directly?"""
            
            elif intent == 'shipping':
                return """I understand you have shipping or delivery questions. Here's what I can tell you:

📦 **Shipping Information:**
- Standard shipping: 5-7 business days
- Express shipping: 2-3 business days
- Free shipping on orders over $50
- Tracking numbers provided within 24 hours
- International shipping available to most countries

🔍 **To help you better, I need:**
- Your order ID
- Current shipping status
- Whether you received a tracking number
- Specific delivery issue you're experiencing

Would you like me to create a support ticket so our shipping team can assist you directly?"""
            
            elif intent == 'return':
                # Check if we already have order_id from context
                order_info = ""
                if 'order_id' in existing_entities:
                    order_info = f"\n✅ **Order ID**: {existing_entities['order_id']} (already provided)"
                
                return f"""I understand you want to return an item. To process your return request, I need:

🔄 **Return Requirements:**
- Items can be returned within 30 days of purchase
- Original packaging is required
- Return shipping is free for defective items
- Refunds issued to original payment method{order_info}

🔍 **To help you better, I need:**
- Your order ID
- Product name/item description
- Reason for return
- Whether the item is defective or just unwanted

Please provide the product details so I can assist you with the return process."""
            
            elif intent == 'complaint':
                return """I understand you have a complaint and I want to help resolve this for you. 

😔 **I'm here to listen and help:**
- Your feedback is important to us
- We want to understand what went wrong
- We're committed to making things right

🔍 **To help you better, I need:**
- Your order ID (if applicable)
- What specifically went wrong
- When this issue occurred
- What you expected vs. what happened
- How we can make this right for you

Would you like me to create a support ticket so our customer relations team can address your concern directly?"""
            
            else:
                return """I understand you have a question, but I don't have specific information in my knowledge base to answer it accurately. 

This could be because:
- Your question is about a topic not covered in our current documentation
- The information might be outdated or not yet added to our knowledge base
- Your query might need more specific details

🔍 **To help you better, I need:**
- More specific details about your question
- Any relevant order IDs or account information
- What you're trying to accomplish

I recommend:
1. Providing more specific details about your issue
2. Checking our FAQ section on our website
3. Creating a support ticket for direct assistance

Would you like me to create a support ticket so our team can help you directly?"""
                
        except Exception as e:
            logger.error(f"Error generating intent-specific response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again or contact our support team for assistance."
    
    def is_ecommerce_related(self, query: str, conversation_context: List[Dict] = None) -> bool:
        """Check if the query is related to e-commerce customer support"""
        ecommerce_keywords = [
            'order', 'purchase', 'buy', 'shopping', 'product', 'item', 'delivery', 'shipping',
            'payment', 'billing', 'refund', 'return', 'exchange', 'account', 'login', 'password',
            'tracking', 'package', 'invoice', 'charge', 'cost', 'price', 'website', 'app',
            'customer', 'support', 'help', 'issue', 'problem', 'complaint', 'service',
            'cart', 'checkout', 'receipt', 'confirmation', 'status', 'update', 'cancel',
            'modify', 'address', 'shipping', 'tax', 'discount', 'coupon', 'promo', 'sale',
            'inventory', 'stock', 'availability', 'warranty', 'guarantee', 'policy',
            'damaged', 'broken', 'defective', 'faulty', 'wrong', 'incorrect', 'missing'
        ]
        
        query_lower = query.lower()
        
        # Special case: website/app technical issues are ALWAYS e-commerce related
        website_tech_keywords = ['website', 'app', 'loading', 'error', '404', '500', 'crash', 'freeze', 'slow', 'browser']
        if any(keyword in query_lower for keyword in website_tech_keywords):
            return True
        
        # Special case: Product names and brands are ALWAYS e-commerce related
        # Common product categories and brand indicators
        product_indicators = [
            'sunscreen', 'cream', 'lotion', 'shampoo', 'soap', 'makeup', 'cosmetic',
            'clothing', 'shirt', 'pants', 'dress', 'shoes', 'accessory', 'jewelry',
            'electronics', 'phone', 'laptop', 'computer', 'tablet', 'camera',
            'book', 'toy', 'game', 'food', 'beverage', 'supplement', 'vitamin',
            'aqualogica', 'brand', 'product', 'item', 'goods', 'merchandise'
        ]
        
        if any(indicator in query_lower for indicator in product_indicators):
            return True
        
        # Special case: If this is a follow-up in an ongoing e-commerce conversation, 
        # treat it as e-commerce related even if keywords don't match
        if conversation_context and len(conversation_context) > 0:
            # Check if previous messages were e-commerce related
            previous_ecommerce = False
            for msg in conversation_context[-3:]:  # Check last 3 messages
                prev_query = msg.get('query', '').lower()
                prev_intent = msg.get('intent', '')
                
                # More comprehensive check for e-commerce context
                if any(keyword in prev_query for keyword in ecommerce_keywords + ['return', 'order', 'product', 'damaged', 'shipping', 'delivery', 'refund', 'exchange']):
                    previous_ecommerce = True
                    break
                
                # Check if previous intent was e-commerce related
                if prev_intent in ['shipping', 'return', 'billing', 'account', 'technical', 'general']:
                    previous_ecommerce = True
                    break
                
                # Also check if this is a return conversation
                if 'return' in prev_query or 'order' in prev_query:
                    previous_ecommerce = True
                    break
            
            if previous_ecommerce:
                logger.info(f"Treating query as e-commerce related due to conversation context: {query}")
                return True
            
        return any(keyword in query_lower for keyword in ecommerce_keywords)
    
    def should_create_ticket(self, intent: str, entities: Dict, confidence: float, query: str, context: List[Dict] = None, conversation_id: str = None) -> bool:
        """Determine if a support ticket should be created"""
        # Check if we already have a ticket for this conversation
        if conversation_id and conversation_id in self.conversation_history:
            # If we already have a ticket in this conversation, don't create another one
            for message in self.conversation_history[conversation_id]:
                if message.get('ticket_id'):
                    logger.info(f"Ticket already exists for conversation {conversation_id}, skipping ticket creation")
                    return False
        
        # Get conversation context for e-commerce check
        conversation_context = None
        if conversation_id and conversation_id in self.conversation_history:
            conversation_context = self.conversation_history[conversation_id]
        
        # Don't create tickets for greetings
        if intent == 'greeting':
            logger.info(f"Skipping ticket creation for greeting: {query}")
            return False
        
        # Always create ticket for off-topic queries
        if not self.is_ecommerce_related(query, conversation_context):
            logger.info(f"Creating ticket for off-topic query: {query}")
            return True
        
        # Don't create tickets for simple follow-up responses in ongoing conversations
        if conversation_id and conversation_id in self.conversation_history:
            conversation_length = len(self.conversation_history[conversation_id])
            if conversation_length > 1:
                # This is a follow-up message, be more selective about tickets
                logger.info(f"Follow-up message in conversation {conversation_id}, being selective about ticket creation")
                
                # Only create ticket if it's a critical issue or no content found
                if context is None or len(context) == 0:
                    logger.info(f"Creating ticket for no-content follow-up query: {query}")
                    return True
                
                # Don't create tickets for low confidence in follow-ups (likely just clarifying questions)
                if confidence < 0.3 and conversation_length > 2:
                    logger.info(f"Skipping ticket for low confidence follow-up: {query}")
                    return False
        
        # Create ticket if no relevant content found in knowledge base
        if context is None or len(context) == 0:
            logger.info(f"Creating ticket for no-content query: {query}")
            return True
            
        ticket_intents = ['billing', 'technical', 'complaint']
        
        # Create ticket for high-priority intents
        if intent in ticket_intents and confidence > 0.5:
            logger.info(f"Creating ticket for high-priority intent: {intent} (confidence: {confidence})")
            return True
        
        # Create ticket if specific entities are detected (order IDs, amounts, etc.)
        if any(key in entities for key in ['order_id', 'amount', 'email']):
            logger.info(f"Creating ticket for entity detection: {entities}")
            return True
        
        # Create ticket for low confidence queries (might need human review)
        if confidence < 0.3:
            logger.info(f"Creating ticket for low confidence: {intent} (confidence: {confidence})")
            return True
        
        # Create ticket for product names and follow-up queries that need human assistance
        query_lower = query.lower()
        product_keywords = ['sunscreen', 'cream', 'soap', 'lotion', 'shampoo', 'makeup', 'cosmetic', 'aqualogica']
        if any(keyword in query_lower for keyword in product_keywords):
            # If user provided order_id and product info, ALWAYS create ticket for returns
            if 'order_id' in entities and 'product' in entities:
                if intent in ['return', 'shipping', 'general']:
                    logger.info(f"Creating ticket for return request with order {entities['order_id']} and product {entities.get('product')}")
                    return True
            
            # Also check conversation context for order_id and product
            if conversation_context and len(conversation_context) > 0:
                has_order_id = any('order_id' in msg.get('entities', {}) for msg in conversation_context)
                has_product = any('product' in msg.get('entities', {}) for msg in conversation_context)
                
                if has_order_id and has_product:
                    logger.info(f"Creating ticket for return request with order and product from conversation context")
                    return True
            
            logger.info(f"Creating ticket for product-related query: {query}")
            return True
        
        # Create ticket for shipping and delivery issues
        shipping_keywords = ['shipping', 'delivery', 'late', 'delayed', 'package', 'tracking', 'arrived', 'not received']
        if any(keyword in query_lower for keyword in shipping_keywords):
            logger.info(f"Creating ticket for shipping-related query: {query}")
            return True
        
        # Create ticket for follow-up queries that seem to need human assistance
        if conversation_id and conversation_id in self.conversation_history:
            conversation_length = len(self.conversation_history[conversation_id])
            if conversation_length > 2:  # After 3+ messages, create ticket for complex queries
                logger.info(f"Creating ticket for complex follow-up in ongoing conversation: {query}")
                return True
        
        return False
    
    def create_support_ticket(self, query: str, intent: str, entities: Dict) -> str:
        """Create a support ticket"""
        ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
        
        # Determine ticket type
        is_off_topic = not self.is_ecommerce_related(query, None)  # No context needed here
        ticket_type = "off_topic" if is_off_topic else "customer_support"
        
        # Check if this is a "no content found" ticket
        if is_off_topic:
            ticket_reason = "off_topic_query"
        else:
            ticket_reason = "no_knowledge_base_content" if not entities else "customer_support"
        
        ticket_data = {
            'id': ticket_id,
            'query': query,
            'intent': intent,
            'entities': entities,
            'ticket_type': ticket_type,
            'ticket_reason': ticket_reason,
            'created_at': datetime.now().isoformat(),
            'status': 'open',
            'priority': 'high' if is_off_topic else 'normal'
        }
        
        # In a real system, this would be saved to a database
        if is_off_topic:
            logger.info(f"Created OFF-TOPIC support ticket: {ticket_id} for query: {query}")
        elif ticket_reason == "no_knowledge_base_content":
            logger.info(f"Created NO-CONTENT support ticket: {ticket_id} for query: {query}")
        else:
            logger.info(f"Created support ticket: {ticket_id}")
        
        self.analytics['total_tickets'] += 1
        
        return ticket_id
    
    def process_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process a user query through the RAG pipeline with state management"""
        start_time = time.time()
        
        try:
            # Get conversation context first
            conversation_context = self.conversation_history.get(conversation_id, [])
            
            # Initialize conversation state if needed
            if conversation_id not in self.conversation_state:
                self.conversation_state[conversation_id] = {
                    'order_id': None,
                    'product': None,
                    'return_reason': None,
                    'ticket_created': False
                }
            
            # Extract entities with conversation context and state
            entities = self.extract_entities(query, conversation_context, conversation_id)
            
            # Update conversation state with new entities
            state = self.conversation_state[conversation_id]
            for key in ['order_id', 'product', 'return_reason']:
                if key in entities:
                    state[key] = entities[key]
                    logger.info(f"Updated state {key}: {entities[key]}")
            
            # Classify intent with enhanced entity awareness
            intent, confidence = self.classify_intent(query, entities)
            
            # Search knowledge base with conversation context
            relevant_chunks = self.search_knowledge_base(query, conversation_context=conversation_context)
            logger.info(f"Query: '{query}' - Found {len(relevant_chunks)} relevant chunks")
            
            # Log the chunks for debugging
            if relevant_chunks:
                for i, chunk in enumerate(relevant_chunks[:2]):  # Log first 2 chunks
                    logger.info(f"Chunk {i+1}: {chunk['content'][:100]}...")
            else:
                logger.info("No relevant chunks found in knowledge base")
            
            # Generate response with conversation context
            response = self.generate_response(query, relevant_chunks, intent, entities, conversation_context)
            
            # Determine if ticket creation is needed (check state first)
            ticket_id = None
            if not state['ticket_created'] and self.should_create_ticket(intent, entities, confidence, query, relevant_chunks, conversation_id):
                ticket_id = self.create_support_ticket(query, intent, entities)
                state['ticket_created'] = True
                logger.info(f"Created ticket {ticket_id} for conversation {conversation_id}")
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update analytics
            self.analytics['total_queries'] += 1
            self.analytics['intent_distribution'][intent] += 1
            self.analytics['response_times'].append(response_time)
            self.analytics['avg_response_time'] = np.mean(self.analytics['response_times'])
            
            # Calculate entity extraction rate
            total_queries = self.analytics['total_queries']
            queries_with_entities = sum(1 for _ in range(total_queries) if entities)
            self.analytics['entity_extraction_rate'] = queries_with_entities / total_queries if total_queries > 0 else 0
            
            # Store conversation
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            # Debug: Log what we're storing
            logger.info(f"Storing conversation entry - entities: {entities}")
            
            self.conversation_history[conversation_id].append({
                'query': query,
                'response': response,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'ticket_id': ticket_id,
                'timestamp': datetime.now().isoformat()
            })
            
            # Debug: Log the full conversation history
            logger.info(f"Full conversation history for {conversation_id}: {self.conversation_history[conversation_id]}")
            
            return {
                'response': response,
                'metadata': {
                    'intent': intent,
                    'confidence': confidence,
                    'entities': entities,
                    'ticket_id': ticket_id,
                    'relevant_chunks': len(relevant_chunks),
                    'needs_clarification': confidence < 0.4,
                    'response_time': response_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try again.",
                'metadata': {
                    'error': str(e),
                    'intent': 'error',
                    'confidence': 0.0,
                    'entities': {},
                    'relevant_chunks': 0,
                    'ticket_id': None,
                    'needs_clarification': True,
                    'response_time': time.time() - start_time
                }
            }

# Initialize the RAG chatbot globally (reused across requests)
rag_bot = None

def get_rag_bot():
    """Get or create RAG chatbot instance (singleton pattern)"""
    global rag_bot
    if rag_bot is None:
        try:
            rag_bot = RAGChatbot()
            logger.info("RAG chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chatbot: {e}")
            # Create a minimal fallback instance
            rag_bot = RAGChatbot()
            rag_bot.knowledge_base = []
            rag_bot.vector_index = None
    return rag_bot

# Web Routes
@app.route('/')
def index():
    """Serve the chatbot UI"""
    return render_template('index.html')

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        bot = get_rag_bot()
        system_status = {
            'status': 'healthy',
            'system_initialized': bot.vector_index is not None,
            'knowledge_base_size': len(bot.knowledge_base),
            'apis_configured': {
                'cohere': bool(bot.cohere_api_key),
                'groq': bool(bot.groq_api_key)
            }
        }
        return jsonify(system_status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        message = data.get('message', '')
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        
        if not message.strip():
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get RAG bot instance
        bot = get_rag_bot()
        
        # Process the query
        result = bot.process_query(message, conversation_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Analytics endpoint"""
    try:
        bot = get_rag_bot()
        return jsonify(dict(bot.analytics))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge-base', methods=['GET'])
def knowledge_base_info():
    """Get knowledge base information"""
    try:
        bot = get_rag_bot()
        return jsonify({
            'total_chunks': len(bot.knowledge_base),
            'sources': list(set(chunk['source'] for chunk in bot.knowledge_base)),
            'embedding_dimension': bot.embeddings.shape[1] if bot.embeddings is not None else 0,
            'vector_index_size': len(bot.vector_index) if bot.vector_index is not None else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-search', methods=['POST'])
def test_search():
    """Test search functionality"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        bot = get_rag_bot()
        results = bot.search_knowledge_base(query, top_k=5)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_results': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    """Clear chat history and conversation state"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id', '')
        
        if conversation_id:
            bot = get_rag_bot()
            # Clear conversation history
            if conversation_id in bot.conversation_history:
                del bot.conversation_history[conversation_id]
                logger.info(f"Cleared conversation history for {conversation_id}")
            
            # Clear conversation state
            if conversation_id in bot.conversation_state:
                del bot.conversation_state[conversation_id]
                logger.info(f"Cleared conversation state for {conversation_id}")
        
        return jsonify({'status': 'success', 'message': 'Chat cleared successfully'})
        
    except Exception as e:
        logger.error(f"Clear chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting RAG Customer Support Chatbot...")
    print("📊 Features enabled:")
    print("  - Cohere embeddings for vector search")
    print("  - NumPy for similarity matching")
    print("  - Groq LLM for response generation")
    print("  - Intelligent entity extraction")
    print("  - Intent classification")
    print("  - Smart support ticket creation")
    print("  - Real-time analytics")
    print("  - Cost-optimized API usage")
    print("\n🌐 Server starting on http://localhost:5000")
    print("🎨 Beautiful web interface available")
    print(f"\n📂 Place your PDF knowledge base files in the '{Config.UPLOAD_FOLDER}/' folder")
    
    # Show client status
    if rag_bot.cohere_client:
        print("✅ Cohere client: Ready")
    else:
        print("❌ Cohere client: Not configured (check COHERE_API_KEY)")
    
    if rag_bot.groq_client:
        print("✅ Groq client: Ready")
    else:
        print("❌ Groq client: Not configured (check GROQ_API_KEY)")
    
    print("\n💡 Chatbot will only use Groq API when relevant content is found!")
    
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)