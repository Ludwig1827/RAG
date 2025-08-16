# RAG + Hybrid Retrieval + Cross-Encoder Reranking (LangChain)

A compact Retrieval-Augmented Generation (RAG) template that:

- Loads **PDF / JSON / CSV / DOCX**
- Splits into chunks with `RecursiveCharacterTextSplitter`
- Builds **dense** (FAISS + OpenAI embeddings) and **sparse** (BM25) retrievers
- Combines them via **Ensemble (hybrid) retrieval**
- (Optional) **Cross-encoders** for reranking (`BAAI/bge-reranker-base`)
- Answers with `RetrievalQA` using an OpenAI chat model  

