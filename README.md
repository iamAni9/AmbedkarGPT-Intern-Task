# AmbedkarGPT - RAG-based Q&A System

An AI-powered command-line Q&A system that answers questions about Dr. B.R. Ambedkar's *"Annihilation of Caste"* using Retrieval-Augmented Generation (RAG).

---

## 📋 Project Overview

This project demonstrates the implementation of a RAG pipeline using:

- **LangChain**: For RAG orchestration  
- **ChromaDB**: Vector database for storing embeddings  
- **HuggingFace Embeddings**: Text-to-vector conversion using `sentence-transformers/all-MiniLM-L6-v2`  
- **Ollama + Mistral 7B**: Free, open-source LLM running locally  

---

## 🛠️ Prerequisites

- Python 3.8 or higher  
- Ollama installed and running locally  
- 4GB+ RAM (for Mistral 7B model)  

---

## 📦 Installation Guide

### 1. Install Ollama and Mistral 7B

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
```
**Windows:**
- Download from [ollama.ai](https://ollama.ai)
- Run the installer
- Open terminal and run:
```bash
ollama pull mistral
```

### 2. Clone/Setup Repository
```bash
mkdir AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
python3 -m venv venv
```
**Activate virtual environment:**<br>
Linux/Mac:
```bash
source venv/bin/activate
```
Windows:
```
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### Terminal 1: Start Ollama Service
```bash
ollama serve
```
This keeps the Ollama LLM service running in the background. Leave this terminal open.<br>

### Terminal 2: Run the Application
Make sure your virtual environment is activated
```bash
python main.py
```
