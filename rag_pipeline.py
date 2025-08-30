import time
import numpy as np
import cohere
import faiss
import json
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, cohere_client, faq_file="faq.json", index_file="faiss_index.bin", 
                 chunks_file="chunks.npy", faq_meta_file="faq_meta.json"):
        self.cohere_client = cohere_client
        self.faq_file = faq_file
        self.index_file = index_file
        self.chunks_file = chunks_file
        self.faq_meta_file = faq_meta_file
        
        self.index = None
        self.chunks = []
        self.faq_data = []
        
        # Try to load existing index, otherwise build it
        try:
            self.load_index()
        except:
            logger.info("Building new FAISS index...")
            self.build_index()
    
    def _embed_with_backoff(self, texts: List[str], input_type: str, max_retries: int = 3) -> Optional[List[List[float]]]:
        """Call Cohere embed with simple exponential backoff."""
        if not self.cohere_client:
            return None
        delay_seconds = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.cohere_client.embed(
                    texts=texts,
                    model="embed-multilingual-v3.0",
                    input_type=input_type
                )
                return resp.embeddings
            except Exception as e:
                logger.warning(f"Embed attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    break
                time.sleep(delay_seconds)
                delay_seconds *= 2
        return None

    def _generate_with_backoff(self, prompt: str, *, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]] = None, max_retries: int = 3) -> Optional[str]:
        """Call Cohere generate with simple exponential backoff and return text."""
        if not self.cohere_client:
            return None
        delay_seconds = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.cohere_client.generate(
                    model="command-r-plus",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences or []
                )
                return resp.generations[0].text.strip()
            except Exception as e:
                logger.warning(f"Generate attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    break
                time.sleep(delay_seconds)
                delay_seconds *= 2
        return None
    
    def build_index(self):
        """Build FAISS index from FAQ data"""
        try:
            # Load FAQ data
            with open(self.faq_file, "r", encoding="utf-8") as f:
                self.faq_data = json.load(f)
            
            # Create chunks from FAQ data
            self.chunks = []
            for entry in self.faq_data:
                q = entry.get("question", "")
                a = entry.get("answer", "")
                chunk = f"Q: {q}\nA: {a}"
                self.chunks.append(chunk)
            
            if not self.chunks:
                logger.error("No chunks created from FAQ data")
                return
            
            # Generate embeddings
            if self.cohere_client:
                embeddings_list = self._embed_with_backoff(self.chunks, input_type="search_document")
                if not embeddings_list:
                    logger.error("Error generating embeddings for index build")
                    return
                embeddings = np.array(embeddings_list).astype("float32")
            else:
                logger.error("Cohere client not available")
                return
            
            # Build FAISS index
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            
            # Save index and metadata
            faiss.write_index(self.index, self.index_file)
            np.save(self.chunks_file, self.chunks)
            
            with open(self.faq_meta_file, "w", encoding="utf-8") as f:
                json.dump(self.faq_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Built FAISS index with {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
    
    def load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            self.index = faiss.read_index(self.index_file)
            self.chunks = np.load(self.chunks_file, allow_pickle=True).tolist()
            
            with open(self.faq_meta_file, "r", encoding="utf-8") as f:
                self.faq_data = json.load(f)
            
            logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def retrieve_chunks(self, query: str, k: int = 3) -> Tuple[str, List[str]]:
        """Retrieve top-k KB chunks and their sources"""
        if not self.index or not self.cohere_client:
            return "", []
        
        try:
            # Get query embedding
            q_emb_list = self._embed_with_backoff([query], input_type="search_query")
            if not q_emb_list:
                raise RuntimeError("Failed to embed query after retries")
            q_emb = q_emb_list[0]
            
            # Search index
            D, I = self.index.search(np.array([q_emb]).astype("float32"), k)
            
            # Get matched chunks and sources
            matched_chunks = []
            sources = []
            
            for idx in I[0]:
                if idx < len(self.chunks):
                    matched_chunks.append(self.chunks[idx])
                    # Get source question as reference
                    if idx < len(self.faq_data):
                        sources.append(self.faq_data[idx]["question"])
                    else:
                        sources.append(f"FAQ Entry {idx}")
            
            return " ".join(matched_chunks), sources
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return "", []
    
    def generate_concise_response(self, user_msg: str, retrieved_text: str) -> str:
        """Generate short actionable answer (2 sentences max)"""
        if not self.cohere_client:
            return self._fallback_response(user_msg, retrieved_text)
        
        try:
            prompt = f"""
You are a concise ecommerce support assistant. Generate EXTREMELY short responses.

User query: {user_msg}
Knowledge: {retrieved_text}

Rules:
1. Maximum 25 words total
2. Only 1-2 sentences
3. Be actionable and direct
4. Use emojis appropriately

Response:"""

            response = self._generate_with_backoff(
                prompt,
                max_tokens=60,
                temperature=0.3,
                stop_sequences=None
            )
            if response is None:
                raise RuntimeError("Failed to generate concise response after retries")
            
            # Fallback to ultra-concise response if still too long
            if len(response.split()) > 25:
                return self._generate_ultra_concise_response(user_msg, retrieved_text)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_msg, retrieved_text)
    
    def _generate_ultra_concise_response(self, user_msg: str, retrieved_text: str) -> str:
        """Generate ultra-concise responses based on patterns"""
        query_lower = user_msg.lower()
        
        if "return" in query_lower or "refund" in query_lower:
            return "🔄 You can return items within 30 days. Need your order number to help further."
        elif "track" in query_lower or "order" in query_lower:
            return "📦 I can help track your order. Please share your order number."
        elif "payment" in query_lower or "billing" in query_lower:
            return "💳 Payment issues can be resolved quickly. What specific problem are you facing?"
        elif "shipping" in query_lower or "delivery" in query_lower:
            return "🚚 Standard shipping takes 3-5 days, express 1-2 days. Free shipping over $50."
        elif "exchange" in query_lower:
            return "🔄 Exchanges available within 30 days. Share your order number to start."
        else:
            return "👋 I'm here to help! Can you provide more details about your question?"
    
    def _fallback_response(self, user_msg: str, retrieved_text: str) -> str:
        """Fallback response when API is unavailable"""
        # Extract first sentence from retrieved text as fallback
        if retrieved_text:
            sentences = retrieved_text.split('. ')
            if sentences:
                return sentences[0] + "."
        return "I can help you with that. Can you provide more details?"
    
    def expand_details(self, retrieved_text: str) -> str:
        """Return full retrieved KB chunk when user asks 'More details'"""
        if not retrieved_text:
            return "I don't have additional details available. Please contact support for more help."
        
        # Format the retrieved text nicely
        formatted_text = retrieved_text.replace("Q: ", "**Question:** ").replace("A: ", "\n**Answer:** ")
        return formatted_text[:1000]  # Cap to avoid giant dumps
    
    def rag_response(self, user_msg: str) -> Tuple[str, str, List[str], float]:
        """Full RAG pipeline: retrieve, generate concise response"""
        start = time.time()
        
        # 1. Retrieve KB chunks
        retrieved, sources = self.retrieve_chunks(user_msg)
        
        # 2. Generate concise response
        if retrieved:
            concise_reply = self.generate_concise_response(user_msg, retrieved)
        else:
            concise_reply = "I can help you with that. Can you provide more details about your question?"
            sources = []
        
        latency = time.time() - start
        return concise_reply, retrieved, sources, latency
    
    def helpful_response(self, user_msg: str) -> Tuple[str, str, List[str], float]:
        """Generate truly helpful, comprehensive responses"""
        start = time.time()
        
        # 1. Retrieve relevant knowledge
        retrieved, sources = self.retrieve_chunks(user_msg)
        
        # 2. Generate helpful, contextual response
        if retrieved:
            helpful_reply = self._generate_helpful_response(user_msg, retrieved)
        else:
            helpful_reply = self._generate_contextual_fallback(user_msg)
            sources = []
        
        latency = time.time() - start
        return helpful_reply, retrieved, sources, latency
    
    def _generate_helpful_response(self, user_msg: str, context: str) -> str:
        """Generate comprehensive, actionable responses"""
        
        # Enhanced prompt for truly helpful responses
        prompt = f"""You are an expert customer support agent. Provide a helpful, comprehensive response that:
1. Directly addresses the customer's question
2. Gives specific, actionable steps
3. Anticipates follow-up questions
4. Shows empathy and understanding
5. Includes relevant details upfront

Keep the tone friendly and professional. Be specific rather than generic.

Context from knowledge base:
{context}

Customer question: {user_msg}

Response:"""

        try:
            helpful_text = self._generate_with_backoff(
                prompt,
                max_tokens=200,
                temperature=0.3,
                stop_sequences=["\n\n"]
            )
            if helpful_text is None:
                raise RuntimeError("Failed to generate helpful response after retries")
            
            # Add contextual enhancements based on query type
            enhanced_response = self._enhance_response(user_msg, helpful_text)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error generating helpful response: {e}")
            return self._generate_contextual_fallback(user_msg)
    
    def _enhance_response(self, user_msg: str, base_response: str) -> str:
        """Add contextual enhancements to make responses more helpful"""
        
        query_lower = user_msg.lower()
        
        # Add specific enhancements based on intent
        if "return" in query_lower or "refund" in query_lower:
            base_response += "\n\n💡 **Quick tip**: Keep your receipt and original packaging for faster processing."
            
        elif "order" in query_lower and "track" in query_lower:
            base_response += "\n\n📱 **Pro tip**: Save our tracking page to your phone's home screen for easy access."
            
        elif "payment" in query_lower or "card" in query_lower:
            base_response += "\n\n🔒 **Security note**: We never store your full card details for your protection."
            
        elif "shipping" in query_lower or "delivery" in query_lower:
            base_response += "\n\n📦 **Good to know**: You'll get SMS/email updates at each delivery milestone."
        
        # Add escalation option for complex issues
        if any(word in query_lower for word in ["problem", "issue", "complaint", "wrong", "error", "help"]):
            base_response += "\n\nIf you need further assistance, I can connect you with our specialist team. Just let me know!"
        
        return base_response
    
    def _generate_contextual_fallback(self, user_msg: str) -> str:
        """Generate contextual responses when no KB match found"""
        
        query_lower = user_msg.lower()
        
        if "refund" in query_lower:
            return ("Here's how refunds work:\n\n"
                    "1️⃣ Returns must be initiated within 30 days\n"
                    "2️⃣ Refunds process in 5-7 business days after we receive items\n"
                    "3️⃣ Original payment method will be credited\n\n"
                    "Need help initiating a return? Just share your order number!")
        
        if "return" in query_lower:
            # Store that this is a return request for context
            return "I'd be happy to help with your return! To get started, I'll need your order number. You can find it in your email confirmation or account dashboard. Returns are typically processed within 30 days of delivery."
            
        elif "order" in query_lower:
            return "I can help you with your order! Please share your order number (starts with #) and I'll look up the current status, tracking info, and estimated delivery date."
            
        elif "payment" in query_lower or "transaction" in query_lower or "failed" in query_lower or "billing" in query_lower:
            if "failed" in query_lower or "error" in query_lower or "declined" in query_lower:
                return "Sorry to hear about the payment issue! Here's how to resolve failed transactions:\n\n✅ **Check**: Card details, expiry date, CVV\n💰 **Verify**: Sufficient balance available\n🏠 **Confirm**: Billing address matches card\n🔄 **Try**: Different payment method if issue persists\n\nIf problems continue, your bank may have flagged the transaction. Contact them or try again in a few minutes."
            else:
                return "I can assist with payment questions! Whether it's about payment methods, failed transactions, or refunds, please let me know the specific issue and I'll guide you through the solution."
            
        elif "shipping" in query_lower:
            return "I'm here to help with shipping questions! We offer standard (3-5 days), express (1-2 days), and same-day delivery in select areas. What specific shipping info do you need?"
            
        else:
            return "I'm here to help! I can assist with orders, returns, payments, shipping, and account questions. What specific issue can I help you resolve today?"
