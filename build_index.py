import json
import numpy as np
import faiss
import cohere
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cohere client
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
if not COHERE_API_KEY:
    print("❌ COHERE_API_KEY not found in environment variables")
    exit(1)

co = cohere.Client(COHERE_API_KEY)

# Input & Output files
FAQ_FILE = "faq.json"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
FAQ_META_FILE = "faq_meta.json"

def build_faiss_index():
    """Build FAISS index from FAQ data"""
    try:
        # Load FAQ data
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        
        print(f"📚 Loaded {len(faq_data)} FAQ entries")
        
        # Create chunks from FAQ data
        chunks = []
        for entry in faq_data:
            q = entry.get("question", "")
            a = entry.get("answer", "")
            chunk = f"Q: {q}\nA: {a}"
            chunks.append(chunk)
        
        print(f"📝 Created {len(chunks)} text chunks")
        
        # Generate embeddings with Cohere
        print("🔄 Generating embeddings...")
        embeddings = co.embed(
            texts=chunks,
            model="embed-multilingual-v3.0",
            input_type="search_document"
        ).embeddings
        
        embeddings = np.array(embeddings).astype("float32")
        print(f"✨ Generated embeddings with dimension {embeddings.shape[1]}")
        
        # Build FAISS index
        print("🏗️ Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # Save index and metadata
        faiss.write_index(index, INDEX_FILE)
        np.save(CHUNKS_FILE, chunks)
        
        with open(FAQ_META_FILE, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully built FAISS index!")
        print(f"📁 Files created:")
        print(f"   - {INDEX_FILE} (FAISS index)")
        print(f"   - {CHUNKS_FILE} (text chunks)")
        print(f"   - {FAQ_META_FILE} (metadata)")
        
        # Test the index
        print("\n🧪 Testing index with sample query...")
        test_query = "How can I return an item?"
        test_embedding = co.embed(texts=[test_query], model="embed-multilingual-v3.0", input_type="search_query").embeddings[0]
        D, I = index.search(np.array([test_embedding]).astype("float32"), k=3)
        
        print(f"Test query: '{test_query}'")
        print("Top matches:")
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(faq_data):
                print(f"  {i+1}. {faq_data[idx]['question']} (distance: {distance:.3f})")
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Building FAISS index from FAQ data...")
    build_faiss_index()
