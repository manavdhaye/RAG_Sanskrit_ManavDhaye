# Sanskrit Hybrid Retrieval-Augmented Generation (RAG) System

A CPU-optimized Sanskrit Retrieval-Augmented Generation (RAG) system designed for Sanskrit document understanding and question answering using Hybrid Retrieval, Indic Embeddings, and Transliteration-Aware Query Processing.

This project supports Sanskrit queries written in both:
- Devanagari script
- Latin transliteration (ITRANS-style Sanskrit)

The system combines:
- Semantic Retrieval using FAISS
- Keyword Retrieval using BM25
- Local Quantized LLM inference
- Indic language-aware preprocessing

to create a lightweight and efficient Sanskrit Question Answering pipeline.

---

# Features

## Sanskrit Document Retrieval
Supports retrieval and question answering over Sanskrit document collections.

## Hybrid Retrieval Architecture
Combines:
- FAISS semantic retrieval
- BM25 keyword retrieval

to improve retrieval quality for morphologically rich Sanskrit text.

## Transliteration-Aware Querying
Supports Sanskrit queries written in Latin script.


CPU-Only Inference

The complete pipeline runs entirely on CPU without GPU acceleration.



Example:

Input:
```text
ghantakarNasya katha kathaya
RAG_Sanskrit_ManavDhaye/
│
├── code/
│   ├── rag_pipeline.py
│   ├── data_loader.py
│   ├── requirements.txt
│
├── data/
│   └── Rag-docs.txt
│
├── models/
│   └── llama-3.2-3b-instruct-q4_k_m.gguf
│
├── report/
│   └── Sanskrit_RAG_Report.pdf
│
├── screenshots/
│
└── README.md

Technologies Used
Component	Technology
Programming Language	Python
Vector Database	FAISS
Sparse Retrieval	BM25
Embedding Model	Vyakyarth Indic Embeddings
LLM Inference	GPT4All / llama.cpp
Transliteration	indic-transliteration
Framework	LangChain
Deployment	CPU-only Local Execution

Installation
Step 1 — Clone Repository
git clone <your_repo_link>
cd RAG_Sanskrit_ManavDhaye

Step 2 — Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Linux / Mac
python3 -m venv venv
source venv/bin/activate

Step 3 — Install Dependencies
pip install -r code/requirements.txt
Required Model

Download a GGUF quantized model and place it inside:
models/
Recommended model:
Llama-3.2-3B-Instruct-Q4_K_M.gguf

Example source:
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

Running the System
python code/rag_pipeline.py

Example Queries

Sanskrit Query
वानराः वने किम् अकुर्वन् ?
Transliteration Query
ghantakarNasya katha kathaya
Another Query
गोवर्धनदासः कथम् कुपितः अभवत् ?
