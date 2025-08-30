# LangChain Memory, RAG & Retrieval Implementation

A comprehensive collection of LangChain implementations covering memory systems, Retrieval-Augmented Generation (RAG), and various retrieval strategies for building production-ready AI applications.

## 🚀 Features

- **Memory Systems**: Multiple conversation memory implementations
- **RAG Pipeline**: Hybrid retrieval with cross-encoder reranking
- **Advanced Retrievers**: Various retrieval strategies and optimizations
- **Document Processing**: Support for PDF, JSON, CSV, and DOCX files
- **Production Ready**: Scalable and configurable components

## 📁 Project Structure

```
├── Memory.ipynb              # Memory systems implementations
├── RAG.py                   # Main RAG class with hybrid retrieval
├── Retriever.ipynb         # Advanced retriever patterns
├── README.md               # This file
└── data/                   # Sample data files
```

## 🧠 Memory Systems

### Available Memory Types

1. **Buffer Memory** - Stores complete conversation history
2. **Buffer Window Memory** - Maintains sliding window of recent messages
3. **Entity Memory** - Tracks entities mentioned in conversations
4. **Knowledge Graph Memory** - Builds relationships between entities
5. **Summary Memory** - Maintains conversation summaries
6. **Vector Store Memory** - Uses embeddings for semantic retrieval
7. **SQLite Memory** - Persistent storage with database backend

### Usage Example

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationBufferMemory(return_messages=True)

# Save conversation context
memory.save_context(
    inputs={"human": "Hello, how are you?"},
    outputs={"ai": "I'm doing well, thank you!"}
)

# Retrieve conversation history
history = memory.load_memory_variables({})['history']
```

## 🔍 RAG Implementation

### Key Features

- **Hybrid Retrieval**: Combines dense (FAISS) and sparse (BM25) retrieval
- **Cross-Encoder Reranking**: Improves result relevance using `BAAI/bge-reranker-base`
- **Multiple File Formats**: PDF, JSON, CSV, DOCX support
- **Configurable Pipeline**: Adjustable weights and parameters

### Quick Start

```python
from RAG import RAG
from langchain_openai import ChatOpenAI

# Initialize RAG system
rag = RAG(
    file_path="your_document.pdf",
    llm=ChatOpenAI(model='gpt-4o-mini')
)

# Create QA chain with hybrid retrieval and reranking
qa = rag.create_chain(
    retriever_type='hybrid',
    rerank=True,
    rerank_top_n=3,
    base_k=10
)

# Ask questions
response = qa.invoke({"query": "What are the key findings?"})
```

### Configuration Options

- **Retriever Types**: `hybrid`, `dense`, `sparse`
- **Reranking**: Enable/disable with custom top-k
- **Chunk Settings**: Configurable size and overlap
- **Weights**: Adjustable dense/sparse retrieval balance

## 🎯 Advanced Retrievers

### Retriever Types Implemented

1. **VectorStore Retriever** - Semantic similarity search
2. **Contextual Compression** - Filters and compresses results
3. **Ensemble Retriever** - Combines multiple retrieval strategies
4. **Long Context Reorder** - Optimizes document order for LLMs
5. **Parent Document Retriever** - Retrieves full documents from chunks
6. **MultiQuery Retriever** - Generates multiple query variations
7. **MultiVector Retriever** - Multiple vectors per document

### Example: Ensemble Retriever

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS

# Create individual retrievers
bm25_retriever = BM25Retriever.from_texts(documents)
faiss_retriever = FAISS.from_texts(documents, embeddings).as_retriever()

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.3, 0.7]  # Favor semantic search
)
```

## 📋 Requirements

```python
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.20
faiss-cpu>=1.7.0
transformers>=4.20.0
torch>=1.12.0
chromadb>=0.4.0
pypdf>=3.0.0
python-docx>=0.8.0
pandas>=1.5.0
numpy>=1.21.0
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-memory-rag-retrieval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
```

## 📖 Usage Examples

### Memory with Conversation Chain

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
```

### RAG with Custom Configuration

```python
# Custom retriever configuration
retriever = rag.create_retriever(
    retriever_type="hybrid",
    rerank=True,
    rerank_top_n=5,
    base_k=20,
    device="cuda"  # Use GPU for reranking
)

# Update ensemble weights
rag.update_weights(dense_weight=0.6, sparse_weight=0.4)
```

### Multi-Document RAG

```python
# Process multiple documents
documents = ["doc1.pdf", "doc2.json", "doc3.csv"]

for doc_path in documents:
    rag = RAG(file_path=doc_path, llm=llm)
    qa = rag.create_chain(retriever_type='hybrid', rerank=True)
    # Process each document...
```

## 🔬 Advanced Features

### Cross-Encoder Reranking

The system uses HuggingFace cross-encoders for improved result relevance:

```python
# Available reranker models
models = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large", 
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
]

# Custom reranker configuration
qa = rag.create_chain(
    rerank=True,
    model_name="BAAI/bge-reranker-large",
    device="cuda",
    rerank_top_n=3
)
```

### Memory Persistence

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Persistent memory with SQLite
def get_chat_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )
```

### Document Processing Pipeline

```python
# Custom chunk processing
def create_custom_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    
    # Add custom metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": f"chunk_{i}",
            "processed_at": datetime.now().isoformat()
        })
    
    return chunks
```

## 📊 Performance Optimization

### GPU Acceleration

```python
# Enable GPU for embeddings and reranking
rag = RAG(
    file_path="document.pdf",
    llm=llm,
    embeddings=OpenAIEmbeddings(),  # Or use local GPU embeddings
)

# GPU reranking
qa = rag.create_chain(
    rerank=True,
    device="cuda",
    model_kwargs={"device": "cuda"}
)
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = ["Question 1?", "Question 2?", "Question 3?"]
responses = []

for query in queries:
    response = qa.invoke({"query": query})
    responses.append(response)
```

## 🛠️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional
TRANSFORMERS_NO_TF=1
KERAS_BACKEND=torch
TF_CPP_MIN_LOG_LEVEL=3
```

### Model Configuration

```python
# LLM Configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000
)

# Embeddings Configuration  
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)
```

## 📈 Monitoring and Logging

```python
import logging

# Enable retrieval logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Custom logging for RAG
logger = logging.getLogger("rag_system")
logger.setLevel(logging.INFO)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## ⚡ Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run example
python -c "from RAG import RAG; from langchain_openai import ChatOpenAI; rag = RAG('sample.pdf', ChatOpenAI()); print('Setup complete!')"
```

---

**Note**: This implementation provides a foundation for building production-ready RAG systems with LangChain. Customize the components based on your specific use case and requirements.

