import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # API Keys
    COHERE_API_KEY = os.getenv('COHERE_API_KEY', 'your_cohere_api_key_here')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Application Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'data'
    
    # RAG Settings
    CHUNK_SIZE = 500
    TOP_K_RESULTS = 3
    MAX_TOKENS = 512
    TEMPERATURE = 0.7
