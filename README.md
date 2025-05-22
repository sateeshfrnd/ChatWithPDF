# ChatWithPDF

A RAG-based chatbot to interact with PDF documents using open-source tools like Python, LangChain, GROQ API for fast LLM inference, and ChromaDB for vector search.


## Features & Architecture 
### Key Features
- Upload and process PDF documents via Streamlit UI
- Split and embed content using LangChain and HuggingFace models
- Store and retrieve embeddings using ChromaDB for fast, local similarity search
- Retrieve relevant document chunks via semantic similarity search
- Generate accurate answers using supported GROQ LLMs (e.g., Llama 3, Gemma)
- Interactive chat interface for querying PDF content
- Supports all models available through the GROQ API
- Retrieval-Augmented Generation (RAG) for context-aware answers
- Persistent vector store for efficient document retrieval across sessions
- GROQ API integration for low-latency, high-performance responses
- Logging for debugging and monitoring

### Architecture

1. **PDF Parsing** : Extract raw text using `PyMuPDF`
2. **Text Chunking** : Split text into manageable chunks with `RecursiveCharacterTextSplitter` from LangChain
3. **Embeddings** : Generate vector embeddings using `HuggingFaceEmbeddings` 
4. **Vector DB** :Store and retrieve embeddings locally with `ChromaDB`
5. **Query Handling** : Retrieve similar chunks using LangChain retriever and apply custom prompts
6. **Answer Generation** : Use GROQ LLMs like (`llama3-70b-8192`, `gemma-9b-it`) for final response
7. **Frontend/UI** : Streamlit-powered UI for chat-based interaction with uploaded documents

## Project Structure
```
├── src/
│   ├── models/
│   │   └── groq_chat_model.py            # Chat model implementation
│   ├── ui/
│   │   ├── chatbot_message_history.py    # wrapper for managing chat message history for Streamlit UI 
│   │   └── chatbot_ui.py                 # Streamlit UI implementation
│   ├── utils/
│   │   ├── constants.py                  # Configuration constants
│   │   └── logger.py                     # Logging
│   ├── vectorstore/
│   │   └── chromadb_store.py             # Persistent vector store
├── app.py                                # Main application entry point
├── requirements.txt                      # Project dependencies
├── test/
│   └── test_logger.py                    # Example test file
└── README.md                             # Project documentation

```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sateeshfrnd/ChatWithPDF.git
cd ChatWithPDF
```

### 2. Create Conda Environment
```
conda create -p chatwithpdf_env python=3.10 -y
conda activate ./chatwithpdf_env
```

### 3. Install Requirements
```
pip install -r requirements.txt
```

> **Note:**  
> If you encounter import errors, try running:
> ```bash
> export PYTHONPATH=src
> ```
> (On Windows: `set PYTHONPATH=src`)


## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the provided local URL
3. Upload a PDF file using the file uploader
4. Start asking questions about the PDF content in the chat interface

## Requirements

- Python 3.10+
- [GROQ API key](https://console.groq.com/keys) (sign up and generate one)
- Internet connection for API access
- The following Python packages (see `requirements.txt`):
    - streamlit
    - langchain-community
    - chromadb
    - langchain-huggingface
    - torch
    - sentence-transformers
    - transformers
    - pymupdf


## Troubleshooting

- **Import Errors:**  
  Ensure your `PYTHONPATH` includes the `src` directory.
- **CUDA/Device Errors:**  
  If you see errors related to PyTorch devices, ensure your system supports CUDA or set device to `"cpu"` in the code.
- **Missing Packages:**  
  Double-check that all dependencies in `requirements.txt` are installed.
