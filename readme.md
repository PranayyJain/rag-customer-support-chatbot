# 🤖 RAG Customer Support Chatbot

A powerful AI-powered customer support chatbot that uses Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware responses based on your knowledge base. The chatbot intelligently balances knowledge base usage with API calls to optimize costs and provide excellent customer support.

## ✨ **Key Features**

### 🧠 **Smart RAG System**
- **PDF Knowledge Base**: Automatically processes and indexes PDF documents
- **Vector Search**: Uses Cohere embeddings for semantic similarity matching
- **FAISS Index**: Fast and efficient similarity search
- **Intelligent Response Generation**: Only uses Groq LLM when relevant content exists

### 🎯 **Smart Query Handling**
- **E-commerce Focus**: Strictly handles e-commerce customer support queries only
- **Automatic Ticket Creation**: Creates support tickets for off-topic queries and missing content
- **Intent Classification**: Automatically detects user intent (billing, technical, account, etc.)
- **Entity Extraction**: Identifies emails, order IDs, amounts, and other entities

### 🎨 **Beautiful Modern UI**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Conversation History**: Full scrollable conversation interface
- **Real-time Analytics**: Track performance metrics and user interactions
- **Support Ticket Management**: Visual feedback for ticket creation

### 💰 **Cost Optimization**
- **Knowledge Base First**: Always searches PDF content before using API
- **Selective API Usage**: Only calls Groq LLM when relevant content is found
- **Automatic Escalation**: Creates tickets instead of expensive API calls for missing content

## 🚀 **Quick Start**

### 1. **Prerequisites**
- Python 3.8+
- Virtual environment (recommended)
- API keys for Cohere and Groq

### 2. **Installation**
```bash
# Clone or download the project
cd rag-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Configuration**
Create a `.env` file in the project root:
```env
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Or update the API keys directly in `app.py`:
```python
self.cohere_api_key = "your_cohere_api_key_here"
self.groq_api_key = "your_groq_api_key_here"
```

### 4. **Knowledge Base Setup**
Place your PDF documents in the `data/` folder:
```
data/
├── your-knowledge-base.pdf
├── customer-support-guide.pdf
└── faq-document.pdf
```

### 5. **Run the Application**
```bash
python app.py
```

The chatbot will be available at `http://localhost:5000`

## 🎯 **How It Works**

### **Query Processing Flow**
```
User Query → E-commerce Check → Knowledge Base Search → Decision Point
                                                      ├─ ✅ Content Found → Use Groq API → Smart Response
                                                      └─ ❌ No Content → Return Message + Create Ticket
```

### **Smart Decision Making**
1. **E-commerce Relevance Check**: Filters out non-e-commerce queries
2. **Knowledge Base Search**: Searches PDF content using vector similarity
3. **Content Availability Check**: Determines if relevant information exists
4. **Response Generation**: Uses API only when content is available
5. **Ticket Creation**: Automatically creates tickets for missing content or off-topic queries

## 📊 **API Endpoints**

### **Web Interface**
- `GET /` - Main chatbot UI

### **API Endpoints**
- `POST /api/chat` - Main chat endpoint
- `GET /api/health` - System health check
- `GET /api/analytics` - Performance analytics
- `GET /api/knowledge-base` - Knowledge base information
- `POST /api/test-search` - Test search functionality

## 🎨 **UI Features**

### **Chat Interface**
- **Message Input**: Character-limited input with real-time counter
- **Conversation History**: Scrollable chat with user/bot message distinction
- **Typing Indicators**: Visual feedback during processing
- **Message Metadata**: Shows intent, confidence, entities, and response time

### **Support Ticket System**
- **Automatic Creation**: Creates tickets for various scenarios
- **Visual Feedback**: Different ticket types with color-coded messages
- **Ticket Information**: Displays ticket ID and next steps

### **Analytics Dashboard**
- **Performance Metrics**: Total queries, tickets, response times
- **Intent Distribution**: Breakdown of query types
- **Entity Extraction**: Success rate of entity detection
- **Real-time Updates**: Live data from conversation history

## 🔧 **Configuration Options**

### **Intent Patterns**
Customize intent detection in `app.py`:
```python
self.intent_patterns = {
    'billing': ['bill', 'charge', 'payment', 'invoice', 'refund'],
    'technical': ['bug', 'error', 'crash', 'not working'],
    'account': ['login', 'password', 'account', 'profile'],
    # Add more intents as needed
}
```

### **Entity Patterns**
Configure entity extraction patterns:
```python
self.entity_patterns = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'order_id': r'\b[A-Z]{2,3}-\d{3,6}\b',
    'amount': r'\$\d+(?:\.\d{2})?',
    # Add more entity types
}
```

### **E-commerce Keywords**
Expand the e-commerce keyword list:
```python
ecommerce_keywords = [
    'order', 'purchase', 'buy', 'shopping', 'product',
    'payment', 'billing', 'shipping', 'delivery',
    # Add more keywords
]
```

## 📁 **Project Structure**
```
rag-chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── SAMPLE_CONVERSATION.md # Example conversation flows
├── data/                 # PDF knowledge base files
├── static/               # CSS, JavaScript, and assets
│   ├── style.css        # Main stylesheet
│   └── script.js        # Frontend functionality
├── templates/            # HTML templates
│   └── index.html       # Main chatbot interface
└── venv/                # Virtual environment (created during setup)
```

## 🧪 **Testing the Chatbot**

### **Test Scenarios**
1. **E-commerce queries with content**: "How do I track my order?"
2. **E-commerce queries without content**: "What's your warranty policy?"
3. **Off-topic queries**: "How do I make a sandwich?"
4. **Queries with entities**: "I was charged $99.99 for order ABC-123"

### **Expected Behaviors**
- **Content Found**: Uses Groq API for intelligent responses
- **No Content**: Returns helpful message + creates ticket
- **Off-Topic**: Redirects + creates ticket
- **High-Priority**: Creates ticket regardless of content availability

## 🔒 **Security Features**

- **Input Validation**: Sanitizes user inputs
- **API Key Protection**: Secure storage of API credentials
- **Rate Limiting**: Built-in request throttling
- **Error Handling**: Graceful error responses without exposing system details

## 🚀 **Deployment**

### **Production Considerations**
- Use production WSGI server (Gunicorn, uWSGI)
- Set `debug=False` in production
- Configure proper logging
- Use environment variables for sensitive data
- Set up monitoring and health checks

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 **License**

This project is open source and available under the MIT License.

## 🆘 **Support**

For issues and questions:
1. Check the sample conversation flow in `SAMPLE_CONVERSATION.md`
2. Review the logs in your terminal
3. Check the analytics dashboard for insights
4. Create an issue in the repository

## 🎉 **What's Next?**

- **Multi-language Support**: Add support for multiple languages
- **Advanced Analytics**: More detailed performance metrics
- **Integration APIs**: Connect with CRM and ticketing systems
- **Mobile App**: Native mobile application
- **Voice Interface**: Speech-to-text and text-to-speech capabilities

---

**Built with ❤️ using Flask, Cohere, FAISS, and Groq**