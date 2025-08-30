# 🛒 Ecommerce Customer Support Chatbot

A sophisticated AI-powered customer support chatbot designed for ecommerce platforms. This chatbot provides intelligent, contextual responses using a combination of AI APIs and a comprehensive knowledge base with RAG (Retrieval-Augmented Generation) capabilities.

## ✨ Features

### 🤖 **AI-Powered Responses**
- **Groq API Integration**: Uses Llama 3 model for natural, contextual conversations
- **Cohere Embeddings**: Semantic search for accurate knowledge base retrieval
- **Fallback System**: Graceful degradation to rule-based responses when APIs are unavailable

### 📚 **Knowledge Base & RAG**
- **Comprehensive Ecommerce Knowledge**: Covers orders, returns, payments, shipping, and more
- **FAISS Vector Search**: Fast similarity search for relevant information
- **TF-IDF Fallback**: Reliable search when embeddings are not available
- **Structured Responses**: Bullet points, emojis, and clear formatting

### 🎯 **Smart Intent Recognition**
- **8 Intent Categories**: Order status, returns, payments, shipping, account management, etc.
- **Entity Extraction**: Automatically detects order numbers, emails, phone numbers
- **Confidence Scoring**: Intelligent response selection based on query analysis

### 💬 **Conversation Management**
- **Session Tracking**: Maintains conversation context across interactions
- **History Storage**: SQLite database for conversation logging
- **Ticket System**: Automated support ticket creation for complex issues

### 🎨 **Modern UI/UX**
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Clean Interface**: Professional chat interface with proper text wrapping
- **Quick Actions**: Pre-defined buttons for common queries
- **Real-time Chat**: Instant responses with typing indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (recommended: Python 3.11)
- pip package manager

### Installation

1. **Clone the repository**
```bash
   git clone <repository-url>
cd customer-support-chatbot
```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   FLASK_ENV=development
   ```

5. **Run the application**
```bash
python app.py
```

6. **Access the chatbot**
   Open your browser and go to `http://localhost:5000`

### Deploy on Render
This repo includes `render.yaml`. Push to GitHub, then on Render choose New > Blueprint and point to your repo. Render will install deps, run gunicorn bound to `$PORT`, and mount a persistent disk at `/data` for database and analytics.

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for AI responses (optional)
- `COHERE_API_KEY`: Your Cohere API key for embeddings
- `PORT`: Server port (default: 5000)
- `DB_PATH`: Path to SQLite DB (default: conversations.db)
- `ANALYTICS_PATH`: Path to analytics JSON (default: analytics.json)

### API Keys Setup
1. **Groq API**: Sign up at [groq.com](https://groq.com) and get your API key
2. **Cohere API**: Sign up at [cohere.ai](https://cohere.ai) and get your API key

## 📱 Usage Examples

### Order Status Query
```
User: "What's the status of my order #12345?"
Bot: 📦 Order #12345 Status

To track your order: 1) Go to 'My Orders' section 2) Enter your order number 3) View real-time tracking status. You'll receive SMS/email updates automatically.

• You'll receive SMS/email updates automatically
• Check 'My Orders' section for real-time tracking
• Contact us if you need immediate assistance
```

### Return Request
```
User: "I want to return an item from order #67890"
Bot: 🔄 Return Request for Order #67890

You can return items within 30 days of delivery. Items must be unused and in original packaging. Refund processing takes 5-7 business days after we receive the returned item.

Return Process:
• Items must be unused and in original packaging
• 30-day return window from delivery date
• Refund processing: 5-7 business days

Next Steps:
• Package item securely
• Include order number #67890
• Use prepaid return label (if provided)
```

## 🏗️ Architecture

### Core Components
- **IntentClassifier**: Categorizes user queries into support categories
- **EntityExtractor**: Extracts relevant information (order numbers, emails, etc.)
- **KnowledgeBase**: Manages ecommerce knowledge and search functionality
- **ConversationManager**: Handles conversation history and session management
- **TicketSystem**: Creates and manages support tickets

### Technology Stack
- **Backend**: Flask (Python web framework)
- **AI/ML**: Groq API, Cohere API, FAISS, scikit-learn
- **Database**: SQLite with conversation tracking
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Search**: Vector similarity search with TF-IDF fallback

## 🔍 How It Works

1. **Query Processing**: User sends a message
2. **Intent Classification**: System determines what the user needs help with
3. **Entity Extraction**: Relevant information (order numbers, etc.) is extracted
4. **Knowledge Search**: RAG system finds relevant information from knowledge base
5. **Response Generation**: AI generates contextual response or uses rule-based fallback
6. **Context Maintenance**: Conversation history is maintained for better responses

## 🛠️ Development

### Project Structure
```
customer-support-chatbot/
├── app.py                 # Main Flask application
├── templates/             # Flask templates
│   └── index.html        # Frontend interface
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── conversations.db      # SQLite database (auto-created)
└── venv/                 # Virtual environment
```

### Adding New Knowledge
Edit the `knowledge_data` list in the `KnowledgeBase.load_knowledge_base()` method to add new support topics.

### Customizing Responses
Modify the `_generate_knowledge_base_response()` method to customize response formats and content.

## 🚨 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'distutils'**
- This is a Python 3.12+ compatibility issue
- The project is configured to work with Python 3.9-3.11
- Use Python 3.11 for best compatibility

**API Errors**
- Check your `.env` file has correct API keys
- Verify API keys are valid and have sufficient credits
- The system will automatically fall back to rule-based responses

**UI Issues**
- Clear browser cache
- Check console for JavaScript errors
- Ensure you're using a modern browser

### Debug Mode
Set `FLASK_ENV=development` in your `.env` file for detailed error messages.

## 📊 Performance

- **Response Time**: < 2 seconds for most queries
- **Concurrent Users**: Supports multiple simultaneous conversations
- **Memory Usage**: Efficient vector search with FAISS
- **Scalability**: Easy to scale with additional knowledge base entries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For technical support or questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the code comments for implementation details

---

**Built with ❤️ for better ecommerce customer support**
