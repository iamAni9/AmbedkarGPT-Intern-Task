from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

class QnASystem:
    def __init__(
        self,
        document_path: str = "speech.txt",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chroma_db_path: str = "./chroma_db",
        llm_model: str = "mistral",
        temperature: float = 0.7,
        top_k_retrieval: int = 3
    ):
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.chroma_db_path = chroma_db_path
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k_retrieval = top_k_retrieval
        
        self.documents = None
        self.chunks = None
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        
        print("✓ QnA System initialized with provided configurations.")
    
    def load_document(self):
        try:
            loader = TextLoader(self.document_path)
            self.documents = loader.load()
            assert len(self.documents) > 0, "No documents loaded."
            print(f"✓ Loaded {len(self.documents)} documents from {self.document_path}")
        except Exception as e:
            print(f"✗ Error loading documents: {e}")
            raise
    
    def split_document_texts(self):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            self.chunks = text_splitter.split_documents(self.documents)
            print(f"✓ Split document into {len(self.chunks)} chunks")
        except Exception as e:
            print(f"✗ Error splitting document: {e}")
            raise
    
    def create_embeddings(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            print(f"✓ Created embeddings using model: {self.embedding_model_name}")
            return self.embeddings
        except Exception as e:
            print(f"✗ Error creating embeddings: {e}")
            raise
    
    def create_vector_store(self):
        try:
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_path,
            )
            print(f"✓ Created vector store at: {self.chroma_db_path}")
        except Exception as e:
            print(f"✗ Error creating vector store: {e}")
            raise
    
    def create_qa_chain(self):
        assert self.vector_store is not None, "Vector store not created."
        
        print(f"\n🤖 Initializing Ollama LLM...")
        print(f"  - Model: {self.llm_model}")
        print(f"  - Temperature: {self.temperature}")
        
        try:
            llm = ChatOllama(
                model=self.llm_model,
                temperature=self.temperature,
                top_p=0.9
            )
            prompt = ChatPromptTemplate.from_template("""
                You are a helpful chatbot. Follow these instructions carefully:
                1. If the question can be answered from the context, answer using ONLY the context.
                2. If the question is greeting or small talk, reply naturally.
                3. If the answer is not in the context, say "I don't know."

                Context:
                {context}

                Question:
                {input}

                Answer:
            """)
            
            print("⛓️ Creating RAG chain...")
            
            doc_chain = create_stuff_documents_chain(llm, prompt)
            self.qa_chain = create_retrieval_chain(
                retriever=self.vector_store.as_retriever(search_kwargs={"k": self.top_k_retrieval}),
                combine_docs_chain=doc_chain
            )

            print("✓ RAG chain created successfully")
        except Exception as e:
            print(f"❌ Error creating QA chain: {str(e)}")
            raise
    
    def answer_question(self, question: str):
        assert self.qa_chain is not None, "QA chain not initialized. Call setup() first."

        if not question.strip():
            return "Question cannot be empty."
        
        try:            
            print("🤖 ", end="")
            for chunk in self.qa_chain.stream({"input": question}):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
            print()
        except Exception as e:
            print(f"❌ Error answering question: {str(e)}")
            raise