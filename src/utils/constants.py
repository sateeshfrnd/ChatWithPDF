"""
Constants and configuration values for the PDF Chat Assistant.
"""
APP_NAME = "ChatWithPDF"

# Default model configuration
DEFAULT_MODEL = "Gemma2-9b-It"

# Default embedding model used to create embeddings for the documents.
DEFULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Available GROQ models and their display names
AVAILABLE_MODELS = {
    "Gemma2-9b-It": "Gemma 2 9B Instruct",
    "llama3-70b-8192": "Llama 3.0 70B",
    "Other": "Other (Enter manually)"
}

# Model selection help text
MODEL_SELECTION_HELP = "Choose the GROQ model to use for chat. Different models have different capabilities and performance characteristics."

# API key configuration
# TODO: Remove this hardcoded API key  
DEFAULT_GROQ_API_KEY = "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"
GROQ_API_KEY_HELP = "Please enter your GROQ API key to use the chat model. Don't have one? Get it from [GROQ](https://console.groq.com/keys)."

# PDF processing configuration
FILE_CHUNK_SIZE = 1000
FILE_CHUNK_OVERLAP = 200 