from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import JSONLoader, CSVLoader, PyPDFLoader, Docx2txtLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # don’t import TF in Hugging Face
os.environ["KERAS_BACKEND"] = "torch"    # keep Keras from choosing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # hide TF C++ INFO/WARN/ERROR logs

from dotenv import load_dotenv
load_dotenv(override=True)

import hashlib


def pretty_print_docs(docs):
    # Accept either a list of Documents OR a RetrievalQA result dict
    if isinstance(docs, dict) and "source_documents" in docs:
        docs = docs["source_documents"]

    print(
        ("\n" + "-" * 100 + "\n").join(
            [f"Document {i+1}:\n\n{d.page_content}" for i, d in enumerate(docs)]
        )
    )


class RAG:
    # file types can be pdf, json, csv, docx
    def __init__(self, file_path, llm, embeddings=OpenAIEmbeddings()):
        self.file_path = file_path                     
        self.content = self._load_content(file_path)
        self.llm = llm
        self.embeddings = embeddings
        self.chunks = self._create_chunks()
        self.dense_retriever = self._create_dense_retriever()
        self.sparse_retriever = self._create_sparse_retriever()
        self.hybrid_retriever = self._create_hybrid_retriever()

    # load different types of docs
    def _load_content(self, file_path):
        if isinstance(file_path, str):
            if file_path.endswith(".json"):
                loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
                docs = loader.load()
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path=file_path, extract_images=False)
                docs = loader.load()
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path=file_path)
                docs = loader.load()
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path=file_path)
                docs = loader.load()
            else:
                raise ValueError("Unsupported file extension")
        else:
            raise ValueError("Unsupported file type or input")
        return docs
    
    # make chunks 
    def _create_chunks(self):
        docs = getattr(self, "content", None)
        if docs is None:
            raise ValueError("self.content is not set")
        if not isinstance(docs, list):                
            docs = [docs]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        chunks = splitter.split_documents(docs)
        # def the metadata
        for i, d in enumerate(chunks):
            d.metadata = {
                **(d.metadata or {}),
                "chunk_idx": i,
                "chunk_id": hashlib.md5((d.page_content[:512] + str(i)).encode()).hexdigest(),
                "source": d.metadata.get("source") or getattr(self, "file_path", "input"),
                "chunk_text": d.page_content
            }
        return chunks
    
    def _create_dense_retriever(self, k=5):
        vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    
    def _create_sparse_retriever(self):
        ret = BM25Retriever.from_documents(self.chunks)  # set k after creation (API-safe)
        ret.k = 5
        return ret
    
    def _create_hybrid_retriever(self, dense_weight=0.5, sparse_weight=0.5):
        return EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[dense_weight, sparse_weight]
        )
    
    # we can also use Cohere
    def _create_reranker(self, top_n=5, model_name: str = "BAAI/bge-reranker-base", **kwargs):
        import torch
        model_kwargs = kwargs.pop("model_kwargs", {}) or {}
        # prefer explicit arg, else autodetect
        if "device" in kwargs:
            model_kwargs["device"] = kwargs.pop("device")
        else:
            model_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        return compressor

    def create_retriever(self, retriever_type="hybrid", rerank=True, rerank_top_n=5, base_k=20, **rerank_kwargs):
        if retriever_type == "hybrid":
            base = self.hybrid_retriever
        elif retriever_type == "dense":
            base = self.dense_retriever
        elif retriever_type == "sparse":
            base = self.sparse_retriever
        else:
            raise ValueError("retriever_type must be 'hybrid', 'dense', or 'sparse'")
        
        if hasattr(base, "search_kwargs"):
            base.search_kwargs["k"] = base_k
        
        if not rerank:
            return base
        
        compressor = self._create_reranker(top_n=rerank_top_n, **rerank_kwargs)
        return ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)


    def create_chain(self, retriever_type='hybrid', rerank: bool = True, rerank_top_n=5, base_k=20, **rerank_kwargs):
        retriever = self.create_retriever(
            retriever_type,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            base_k=base_k,
            **rerank_kwargs
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True    
        )
    
    def update_weights(self, dense_weight=0.5, sparse_weight=0.5):
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[dense_weight, sparse_weight]
        )

# use case
if __name__ == "__main__":
    rag = RAG(file_path="people.json", llm=ChatOpenAI(model='gpt-4o-mini'))
    qa = rag.create_chain(
        retriever_type='hybrid',          
        rerank=True,
        rerank_top_n=3,
        base_k=5,
        # device='cpu'                       
    )
    docs = qa.invoke({"query": "List all the people's information with name starts at 'A', 'B' or 'C'." })
    pretty_print_docs(docs)
