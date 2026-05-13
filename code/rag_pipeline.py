import os
import re

# Transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
# Langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain.retrievers.ensemble import EnsembleRetriever  # <-- THIS IS THE UPDATED LINE
# from langchain_community.retrievers import EnsembleRetriever
# from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# from langchain.messages import AIMessage

# Custom Loader
from data_loader import load_custom_sanskrit_docs

class SanskritHybridRAG:
    def __init__(self, data_path: str, model_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.input_was_latin = False
        
        print("--- Initializing Sanskrit RAG Pipeline (CPU Optimized) ---")
        self.documents = load_custom_sanskrit_docs(data_path)
        if not self.documents:
            raise ValueError("No documents loaded. Please check your data file.")
            
        self.retriever = self._setup_hybrid_retriever()
        self.llm = self._setup_cpu_llm()
        self.qa_chain = self._setup_qa_chain()

    def _is_latin(self, text: str) -> bool:
        """Detects if the input text contains mostly Latin script."""
        return bool(re.search(r'[a-zA-Z]', text))

    def _transliterate_to_devanagari(self, text: str) -> str:
        """Converts user's Latin input (ITRANS) to Devanagari."""
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)

    def _transliterate_to_latin(self, text: str) -> str:
        """Converts Devanagari back to Latin for output consistency."""
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)

    def _setup_hybrid_retriever(self):
        print("Setting up Hybrid Retriever...")
        
        # 1. Dense Retriever (Vyakyarth)
        print(" > Loading HuggingFace Embeddings (Vyakyarth)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="krutrim-ai-labs/Vyakyarth",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(self.documents, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 2. Sparse Retriever (BM25)
        print(" > Initializing BM25 Sparse Retriever...")
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = 3

        # 3. Ensemble (Hybrid)
        print(" > Combining into Ensemble Retriever (60% Dense, 40% Sparse)...")
        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6]
        )

    def _setup_cpu_llm(self):
        print(f"Loading Quantized LLM via llama.cpp: {self.model_path}")
        return GPT4All(
            model=self.model_path,
            max_tokens=2048,
            temp=0.1
        )

    def _setup_qa_chain(self):
        # The prompt explicitly asks the LLM to use the source IDs
        template = """
            You are a Sanskrit Question Answering Assistant.

            Answer ONLY using the provided Sanskrit context.

            STRICT RULES:
            1. Respond ONLY in Sanskrit.
            2. Do NOT use Marathi, Hindi, or English.
            3. Do NOT repeat sentences.
            4. Keep the answer within 2-3 sentences.
            5. If answer is not found, reply:
            'न ज्ञायते'
            6. Do NOT explain.
            7. Do NOT add notes.

            Context:
            {context}

            Question:
            {input}

            Concise Sanskrit Answer:
            """
        
        prompt = PromptTemplate(template=template, input_variables=["context", "input"])
        # Combine documents into the prompt
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create the final retrieval chain
        return create_retrieval_chain(self.retriever, combine_docs_chain)

    def query(self, user_input: str):
        # Step 1: Detect Script and Transliterate if needed
        self.input_was_latin = self._is_latin(user_input)
        processed_query = user_input
        
        if self.input_was_latin:
            print("\n[Input Gateway] Latin script detected. Transliterating to Devanagari...")
            processed_query = self._transliterate_to_devanagari(user_input)
            print(f" >> Query as Devanagari: {processed_query}")
        
        # Step 2: Retrieve and Generate
        print(f"\n[Retrieving Context & Generating Response...]")
        response = self.qa_chain.invoke({"input": processed_query})
        
        result = response["answer"]
        sources = [doc.metadata['source_id'] for doc in response['context']]

        # Step 3: Post-process output
        if self.input_was_latin:
            print("[Output Gateway] Converting response back to Latin script...")
            result = self._transliterate_to_latin(result)
        
        print("\n=== RESPONSE ===")
        print(result)
        print(f"=== Sources Used: {sources} ===")
        return result

if __name__ == "__main__":
    # --- Configuration ---
    DATA_FILE = "data/Rag-docs.txt" 
    
    # NOTE: You MUST download a .gguf model and place it here.
    # Example: Llama-3.2-3B-Instruct-Q4_K_M.gguf from HuggingFace (GGUF format)
    MODEL_FILE = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: You must download a quantized GGUF model and place it at: {MODEL_FILE}")
        print("Try: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
    else:
        # Initialize RAG
        rag = SanskritHybridRAG(DATA_FILE, MODEL_FILE)
        
        # Test 1: Devanagari Query
        print("\n-----------------------------------------")
        rag.query("वानराः वने किम् अकुर्वन् ?") # "What did the monkeys do in the forest?"

        # Test 2: Latin Query (ITRANS)
        print("\n-----------------------------------------")
        rag.query("govardhanadaasaH katham kupitaH abhavat?") # "Why did Govardhandas get angry?"