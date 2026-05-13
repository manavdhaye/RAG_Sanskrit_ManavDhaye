# Sanskrit Hybrid Retrieval-Augmented Generation (RAG) System

A CPU-optimized Sanskrit Retrieval-Augmented Generation (RAG) system designed for Sanskrit document understanding and question answering using Hybrid Retrieval, Indic Embeddings, and Transliteration-Aware Query Processing.

The system supports Sanskrit queries written in both:

- Devanagari Script
- Latin Transliteration (ITRANS-style Sanskrit)

The architecture combines:

- Semantic Retrieval using FAISS
- Keyword Retrieval using BM25
- Local Quantized LLM Inference
- Indic Language-Aware Preprocessing

to create an efficient and lightweight Sanskrit Question Answering pipeline.

---

# Features

## Sanskrit Document Retrieval
Supports retrieval and question answering over Sanskrit document collections.

---

## Hybrid Retrieval Architecture

The system combines:

- FAISS Semantic Retrieval
- BM25 Keyword Retrieval

to improve retrieval quality for morphologically rich Sanskrit text.

---

## Transliteration-Aware Querying

Supports Sanskrit queries written in Latin transliteration.

Example:

### Input
```text
ghantakarNasya katha kathaya
```

### Internal Normalization
```text
घण्टाकर्णस्य कथा कथय
```

This improves:
- semantic retrieval
- keyword matching
- Sanskrit consistency

---

## CPU-Only Inference

The complete pipeline runs entirely on CPU without GPU acceleration using a quantized GGUF-based LLM.

---

# System Architecture

```text
User Query
    ↓
Language / Transliteration Detection
    ↓
Devanagari Normalization
    ↓
Hybrid Retrieval
(FAISS + BM25)
    ↓
Context Construction
    ↓
CPU-based Quantized LLM
    ↓
Sanskrit Response
```

---

# Project Structure

```text
RAG_Sanskrit_ManavDhaye/
│
├── code/
│   ├── rag_pipeline.py
│   ├── data_loader.py
│   └── requirements.txt
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
```

---

# Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Vector Database | FAISS |
| Sparse Retrieval | BM25 |
| Embedding Model | Vyakyarth Indic Embeddings |
| LLM Inference | GPT4All / llama.cpp |
| Transliteration | indic-transliteration |
| Framework | LangChain |
| Deployment | CPU-only Local Execution |

---

# Installation

## Step 1 — Clone Repository

```bash
git clone <your_repo_link>
cd RAG_Sanskrit_ManavDhaye
```

---

## Step 2 — Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r code/requirements.txt
```

---

# Required Model

Download a GGUF quantized model and place it inside:

```text
models/
```

Recommended Model:

```text
llama-3.2-3b-instruct-q4_k_m.gguf
```

Example Source:

[Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF?utm_source=chatgpt.com)

---

# Running the System

```bash
python code/rag_pipeline.py
```

---

# Example Queries

## Sanskrit Query

```text
वानराः वने किम् अकुर्वन् ?
```

---

## Transliteration Query

```text
ghantakarNasya katha kathaya
```

---

## Another Sanskrit Query

```text
गोवर्धनदासः कथम् कुपितः अभवत् ?
```

---

# Sample Output

## Query

```text
गोवर्धनदासः कथम् कुपितः अभवत् ?
```

## Generated Response

```text
tena virUpastrI mukhe dRRiDhatvA sthitA, pratyAgachChati, AtmAnnindaya, lalATahastena tADayan cha |
```

## Sources Used

```text
[5, 4, 19, 28, 2]
```

---

# Current Limitations

- General-purpose LLM occasionally generates mixed-language responses
- Sanskrit generation quality can be improved using Sanskrit fine-tuned models
- Long generations may occasionally introduce repetition artifacts

Despite these limitations, the retrieval system successfully demonstrates:
- Sanskrit-aware retrieval
- transliteration normalization
- hybrid semantic + keyword retrieval
- CPU-only local inference

---

# Future Improvements

- Cross-encoder reranking
- Sanskrit fine-tuned LLMs
- Multilingual query support
- Agentic retrieval systems
- Advanced Sanskrit NLP preprocessing

---

# Author

Manav Dhaye
