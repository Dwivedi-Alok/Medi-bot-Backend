from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import traceback
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend (all origins, for dev)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

config = Config()

# Global variables for caching
vectorstore = None
qa_chain = None

def initialize_vectorstore():
    """Initialize and cache the vector store."""
    global vectorstore
    try:
        logger.info("Loading vector store...")
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-MiniLM-L3-v2'
        )
        vectorstore = FAISS.load_local(
            config.DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        return False

def initialize_qa_chain():
    """Initialize and cache the QA chain."""
    global qa_chain
    try:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        logger.info("Initializing QA chain...")
        
        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        You are MediBot, a professional medical AI assistant. Answer the question using only the information provided in the context.
        
        Guidelines:
        - Provide clear, accurate medical information based on the context
        - Be professional, empathetic, and concise
        - Use bullet points for better readability when listing multiple items
        - If the answer is not found in the context, respond with "I don't have specific information about this in my current knowledge base."
        - Always remind users to consult healthcare professionals for medical decisions
        - Highlight important medical terms naturally in your response
        
        Context: {context}
        Question: {question}
        
        Professional Medical Response:
        """
        
        prompt_template = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name=config.MODEL_NAME,
                temperature=0.1,
                groq_api_key=config.GROQ_API_KEY,
                max_tokens=1000
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt_template}
        )
        
        logger.info("QA chain initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize QA chain: {str(e)}")
        return False

def highlight_medical_terms(text):
    """Highlight important medical terms in the response."""
    medical_terms = [
        "symptoms", "diagnosis", "treatment", "medication", "dosage", 
        "side effects", "fever", "infection", "inflammation", "pain",
        "blood pressure", "heart rate", "diabetes", "hypertension", 
        "antibiotics", "dehydration", "nausea", "headache", "fatigue"
    ]
    
    highlighted_text = text
    for term in medical_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(lambda m: f"**{m.group()}**", highlighted_text)
    
    return highlighted_text

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint for medical queries."""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Invalid request. 'message' field is required."
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                "error": "Message cannot be empty."
            }), 400
        
        logger.info(f"Received query: {user_message[:100]}...")
        
        if not vectorstore or not qa_chain:
            return jsonify({
                "error": "Medical knowledge base is not available. Please try again later."
            }), 503
        
        response = qa_chain.invoke({'query': user_message})
        result = response.get("result", "")
        source_documents = response.get("source_documents", [])
        
        enhanced_result = highlight_medical_terms(result)
        
        enhanced_result += "\n\n---\n⚠️ **Medical Disclaimer:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment."
        
        sources = []
        for i, doc in enumerate(source_documents[:3], 1):
            page_info = doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))
            content_preview = doc.page_content.strip()[:200]
            if len(doc.page_content) > 200:
                content_preview += "..."
            sources.append({
                "id": i,
                "page": page_info,
                "content": content_preview
            })
        
        response_data = {
            "response": enhanced_result,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "query_processed": True
        }
        
        logger.info("Query processed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "I'm experiencing technical difficulties. Please try again in a moment.",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    suggestions = [
        "What are the symptoms of dehydration?",
        "Tell me about blood pressure medications",
        "How to treat a common cold naturally?",
        "What causes persistent headaches?",
        "Explain the side effects of antibiotics",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "How to manage fever in adults?",
        "What are the signs of a heart attack?"
    ]
    return jsonify({
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "system_status": "operational",
        "vectorstore_loaded": vectorstore is not None,
        "qa_chain_loaded": qa_chain is not None,
        "model_name": config.MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/api/health",
            "/api/chat",
            "/api/suggestions",
            "/api/stats"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

# --- Initialization for all environments (local and Render) ---
print("🩺 MediBot API Server Starting...")
print("=" * 50)

print("📚 Loading medical knowledge base...")
if not initialize_vectorstore():
    print("❌ Failed to load vector store. Exiting.")
    exit(1)

print("🧠 Initializing AI model...")
if not initialize_qa_chain():
    print("❌ Failed to initialize QA chain. Exiting.")
    exit(1)

print("✅ All services initialized successfully!")
print("=" * 50)
# -------------------------------------------------------------

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("=" * 50)
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
