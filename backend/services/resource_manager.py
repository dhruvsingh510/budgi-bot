import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv


class ResourceManager:
    """Centralized resource management for the application."""

    def __init__(self):
        load_dotenv()
        self._text_splitter = None
        self._embeddings = None
        self._budget_vector_store = None
        self._transaction_vector_store = None
        self._llm = None

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get or create text splitter (singleton)."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100, chunk_overlap=10
            )
        return self._text_splitter

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embeddings model (singleton)."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    @property
    def llm(self) -> ChatGroq:
        """Get or create LLM (singleton)."""
        if self._llm is None:
            self._llm = ChatGroq(
                model_name=os.environ.get("LITELLM_MODEL"),
                groq_api_key=os.environ.get("GROQ_API_KEY"),
            )
        return self._llm

    def get_budget_vector_store(self) -> FAISS:
        """Get or create budget FAISS store (singleton)."""
        if self._budget_vector_store is None:
            faiss_path = self._get_budget_faiss_path()
            if os.path.exists(f"{faiss_path}/index.faiss"):
                self._budget_vector_store = FAISS.load_local(
                    faiss_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("✅ Loaded budget FAISS store")
            else:
                print("🔴 Budget FAISS store not found")
        return self._budget_vector_store

    def get_transaction_vector_store(self) -> FAISS:
        """Get or create transaction FAISS store (singleton)."""
        if self._transaction_vector_store is None:
            faiss_path = self._get_transaction_faiss_path()
            if os.path.exists(f"{faiss_path}/index.faiss"):
                try:
                    self._transaction_vector_store = FAISS.load_local(
                        faiss_path,
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    print("✅ Loaded transaction FAISS store")
                except Exception as e:
                    print(f"🔴 Failed to load transaction FAISS store: {e}")
            else:
                print("🔴 Transaction FAISS store not found")
        return self._transaction_vector_store

    def _get_budget_faiss_path(self) -> str:
        """Get path to budget FAISS store."""
        return os.path.abspath(
            os.path.join(Path(__file__).parent, "..", "data", "faiss_store_budget")
        )

    def _get_transaction_faiss_path(self) -> str:
        """Get path to transaction FAISS store."""
        return os.path.abspath(
            os.path.join(Path(__file__).parent, "..", "data", "faiss_store_transaction")
        )


# Global instance (singleton pattern)
resource_manager = ResourceManager()
