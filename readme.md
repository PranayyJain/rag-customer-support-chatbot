# 🤖 RAG Customer Support Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/yourusername/rag-customer-support-chatbot)

A powerful AI-powered customer support chatbot that uses Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware responses based on your knowledge base. The chatbot intelligently balances knowledge base usage with API calls to optimize costs and provide excellent customer support.

## 🚀 **Live Demo**

- **Demo URL**: [Coming Soon]
- **Documentation**: [Full Documentation](https://github.com/yourusername/rag-customer-support-chatbot/wiki)
- **Issues**: [Report Issues](https://github.com/yourusername/rag-customer-support-chatbot/issues)

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

### 💰 **Cost Optimization & Performance**

### **Smart API Usage**

- **Knowledge Base First**: Always searches PDF content before using API
- **Selective API Usage**: Only calls Groq LLM when relevant content is found
- **Automatic Escalation**: Creates tickets instead of expensive API calls for missing content
- **Response Caching**: Intelligent caching to reduce repeated API calls

### **Performance Features**

- **Fast Vector Search**: FAISS index for sub-second response times
- **Efficient Chunking**: Optimal text chunking for better search results
- **Memory Optimization**: Smart memory management for large knowledge bases
- **Async Processing**: Non-blocking operations for better user experience

### **Cost Breakdown**

- **Cohere Embeddings**: ~$0.10 per 1M tokens (one-time cost for knowledge base)
- **Groq LLM**: ~$0.05 per 1M tokens (only when generating responses)
- **Estimated Monthly Cost**: $5-20 depending on usage (much cheaper than traditional chatbots)

## 🚀 **Quick Start**

### 1. **Prerequisites**

- Python 3.8+
- Virtual environment (recommended)
- API keys for Cohere and Groq

### 2. **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-customer-support-chatbot.git
cd rag-customer-support-chatbot

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

#### **Option A: Environment Variables (Recommended)**

Create a `.env` file in the project root:

```env
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

#### **Option B: Direct Configuration**

Update the API keys directly in `config.py`:

```python
COHERE_API_KEY = 'your_cohere_api_key_here'
GROQ_API_KEY = 'your_groq_api_key_here'
```

#### **Option C: System Environment Variables**

Set environment variables in your system:

```bash
# Windows
set COHERE_API_KEY=your_key_here
set GROQ_API_KEY=your_key_here

# macOS/Linux
export COHERE_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here
```

### 4. **Get API Keys**

#### **Cohere API Key**

1. Sign up at [cohere.ai](https://cohere.ai)
2. Go to your dashboard
3. Copy your API key
4. Add to your configuration

#### **Groq API Key**

1. Sign up at [groq.com](https://groq.com)
2. Navigate to API keys section
3. Generate a new API key
4. Add to your configuration

### 5. **Knowledge Base Setup**

Place your PDF documents in the `data/` folder:

```
data/
├── your-knowledge-base.pdf
├── customer-support-guide.pdf
└── faq-document.pdf
```

### 6. **Run the Application**

```bash
python app.py
```

The chatbot will be available at `http://localhost:5000`

### 7. **Test the Chatbot**

1. Open your browser and go to `http://localhost:5000`
2. Try asking questions like:
   - "How do I track my order?"
   - "What's your return policy?"
   - "I need help with billing"
3. Check the analytics dashboard to see performance metrics

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

## 🎨 **UI Features & Screenshots**

### **Chat Interface**

- **Message Input**: Character-limited input with real-time counter
- **Conversation History**: Scrollable chat with user/bot message distinction
- **Typing Indicators**: Visual feedback during processing
- **Message Metadata**: Shows intent, confidence, entities, and response time

### **Key UI Components**

- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Modern Interface**: Clean, professional design with smooth animations
- **Real-time Updates**: Live chat experience with instant responses
- **Accessibility**: High contrast and readable fonts for all users

### **Screenshots**

*Screenshots will be added here once the project is deployed*

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
rag-customer-support-chatbot/
├── app.py                 # Main Flask application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── PROJECT_STRUCTURE.md  # Detailed project structure
├── .gitignore            # Git ignore rules
├── data/                 # PDF knowledge base files
│   └── README.md        # Data directory instructions
├── static/               # CSS, JavaScript, and assets
│   ├── style.css        # Main stylesheet
│   └── script.js        # Frontend functionality
└── templates/            # HTML templates
    └── index.html       # Main chatbot interface
```

For detailed information about each file and component, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

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

## 🚀 **Deployment & Hosting**

### **🌐 Free Hosting Recommendations**

#### **1. Render (Recommended - Fastest)**

- **Free Tier**: 750 hours/month
- **Pros**: Auto-deploy from GitHub, custom domains, SSL included
- **Setup**: Connect GitHub repo → Auto-deploy
- **URL**: [render.com](https://render.com)

#### **2. Railway**

- **Free Tier**: $5 credit/month
- **Pros**: Very fast, easy deployment, good performance
- **Setup**: GitHub integration with automatic deployments
- **URL**: [railway.app](https://railway.app)

#### **3. Fly.io**

- **Free Tier**: 3 shared-cpu VMs, 3GB persistent volume
- **Pros**: Global edge deployment, excellent performance
- **Setup**: CLI-based deployment
- **URL**: [fly.io](https://fly.io)

#### **4. Heroku (Alternative)**

- **Free Tier**: Discontinued, but paid plans start at $7/month
- **Pros**: Easy deployment, good ecosystem
- **Setup**: GitHub integration
- **URL**: [heroku.com](https://heroku.com)

### **🚀 Quick Deploy to Render**

1. **Fork this repository** to your GitHub account
2. **Sign up** at [render.com](https://render.com)
3. **Connect your GitHub** account
4. **Create New Web Service**
5. **Select your forked repository**
6. **Configure environment variables**:
   ```
   COHERE_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   FLASK_ENV=production
   FLASK_DEBUG=False
   ```
7. **Deploy** - Your chatbot will be live in minutes!

### **🐳 Docker Deployment**

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### **📋 Production Checklist**

- [ ] Set `FLASK_ENV=production`
- [ ] Set `FLASK_DEBUG=False`
- [ ] Use production WSGI server (Gunicorn)
- [ ] Configure environment variables
- [ ] Set up monitoring and health checks
- [ ] Enable SSL/HTTPS
- [ ] Set up custom domain (optional)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 **License**

This project is open source and available under the MIT License.

## 🆘 **Support & Troubleshooting**

### **Common Issues & Solutions**

#### **1. API Key Errors**

```
Error: Invalid API key for Cohere/Groq
Solution: Verify your API keys in .env file or config.py
```

#### **2. PDF Processing Issues**

```
Error: No PDF files found
Solution: Ensure PDFs are in the data/ folder and contain extractable text
```

#### **3. Memory Issues**

```
Error: Out of memory during indexing
Solution: Reduce PDF file sizes or split large documents
```

#### **4. Slow Response Times**

```
Issue: Chatbot responds slowly
Solution: Check internet connection and API service status
```

### **Getting Help**

1. **Check Logs**: Review terminal output for error messages
2. **Analytics Dashboard**: Use the built-in analytics to diagnose issues
3. **GitHub Issues**: [Create an issue](https://github.com/yourusername/rag-customer-support-chatbot/issues) for bugs
4. **Discussions**: [Start a discussion](https://github.com/yourusername/rag-customer-support-chatbot/discussions) for questions
5. **Wiki**: Check the [project wiki](https://github.com/yourusername/rag-customer-support-chatbot/wiki) for detailed guides

## 🎉 **Roadmap & Future Features**

### **🚀 Coming Soon (v2.0)**

- **Multi-language Support**: Spanish, French, German, and more
- **Advanced Analytics**: Detailed performance metrics and insights
- **Integration APIs**: Connect with Zendesk, Intercom, and other CRM systems
- **Voice Interface**: Speech-to-text and text-to-speech capabilities
- **Mobile App**: Native iOS and Android applications

### **🔮 Future Versions (v3.0+)**

- **AI Training**: Learn from customer interactions to improve responses
- **Multi-modal Support**: Handle images, documents, and video
- **Enterprise Features**: SSO, role-based access, audit logs
- **Advanced NLP**: Better understanding of complex queries
- **Real-time Collaboration**: Multiple agents working together

### **💡 Feature Requests**

Have an idea for a feature? [Create a feature request](https://github.com/yourusername/rag-customer-support-chatbot/issues/new?template=feature_request.md)!

### **🤝 Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

---

**Built with ❤️ using Flask, Cohere, FAISS, and Groq**

---

## 🔑 **Important Security Note**

The API keys in the code have been replaced with placeholder values for security. Before running the application:

1. **Get your API keys** from [Cohere](https://cohere.ai) and [Groq](https://groq.com)
2. **Create a `.env` file** in the project root with your actual keys:
   ```env
   COHERE_API_KEY=your_actual_cohere_key_here
   GROQ_API_KEY=your_actual_groq_key_here
   ```
3. **Never commit API keys** to version control

This ensures your API keys remain secure and private.
