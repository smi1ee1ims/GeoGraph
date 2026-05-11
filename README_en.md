# GeoAgent - Your Personalized Intelligent Geography Question Generator

[中文](./README.md) | [English](./README_en.md)

Live demo: http://47.111.172.69/

For questions, feedback, or collaboration: 1345119185@qq.com

### ⏫ Updates - 2026.5.11
- Interactive agent for refining generated questions
- Built-in evaluator and optimizer
- Redesigned UI for better usability

### 【1】Overview

GeoAgent is a multi-route RAG-based **personalized** geography question generator with two core strengths:

1. **Proprietary Knowledge Base**: Supports three generation modes — **Graph Retrieval**, **Vector Retrieval**, and **No Retrieval**

2. **Personalized Customization**: Dynamically adjusts question difficulty and cognitive level based on student group profiles

<img width="1893" height="884" alt="image" src="https://github.com/user-attachments/assets/2e013195-2175-43a7-ab4a-8feca727e155" />

Powered by deepseek-chat-V3.2 and qwen-max.

Currently supports Chapter 2 for testing. More chapters coming soon.

### 【2】Core Features

- **Graph Retrieval Generation**: Uses Neo4j knowledge graph for relational path retrieval
- **Vector Retrieval Generation**: Uses FAISS semantic vector store for similarity search
- **No Retrieval Generation**: Direct LLM generation based on knowledge points
- **Adaptive Question Generation**: Fuses question bank feature model with student profiles to dynamically adjust:
  - Difficulty level (Easy / Medium / Hard)
  - Cognitive level (Remember → Understand → Apply → Analyze → Evaluate → Create)
  - Calculation probability

### 【3】Tech Stack
- Frontend: HTML, jQuery
- Backend: FastAPI
- Agent chain: LangChain
- Database: Neo4j, AuraDB, VectorStore
- LLM API: deepseek-chat-v3.2, qwen-max

### 【4】Project Structure

Only common files are listed. Others are generally not needed.

```
script/     # Local development version
├── graphRAG.ipynb # Question generation
├── eval.ipynb  # Evaluators 1-3
├── eval_structural_similarity.ipynb  # Evaluator 4
├── question_optimizer_agent.py  # Question optimizer agent
backend/    # Web service version
├── main.py              # FastAPI entry point
├── rag_service.py      # Core RAG service
├── build_vectorstore.py # Vector store builder
├── index.html          # Frontend page
├── requirements.txt    # Dependencies
├── .env                # Environment variables
└── data/
    ├── cleaned_湘教版地理必修一Chap2.md  # Textbook content
    └── vectorstore/    # FAISS vector store
```

### 【5】Quick Start Guide

`sript/` and `backend/` are the local development version and web service version respectively. Either can implement full functionality. Each has its own `.env` that needs separate configuration.
The `data/` folder is required.

Using backend version as example:
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Configure environment variables (.env)
You can also use other LLMs, but may need to modify code elsewhere.
For script version:
```bash
DEEPSEEK_API_KEY=your_key
QWEN_KEY=your_key
```
3. Build vector store (first run only)
```bash
python build_vectorstore.py
```
4. Start the service
```bash
python main.py
```
Access `http://localhost` to use the frontend.
Or visit http://47.111.172.69/ for the deployed server version.

### 【6】Core Scripts (without frontend)

#### graphRAG.ipynb - Question Generation
Run cells directly to generate questions, supports three modes:
- **Graph Retrieval**: Uses Neo4j knowledge graph
- **Vector Retrieval**: Uses FAISS vector store
- **No Retrieval**: Direct LLM generation
Also includes personalized feature fusion module and question feature dictionary.

#### eval.ipynb - Evaluators 1-3
Three evaluation metrics, run corresponding cells, supports single and batch evaluation:
- Difficulty evaluation
- Cognitive level evaluation
- Knowledge point coverage evaluation

#### eval_structural_similarity.ipynb - Evaluator 4
Structural similarity evaluation, comparing generated questions with the original question bank.

#### question_optimizer_agent.py - Question Optimizer Agent
Run in terminal for interactive question optimization:

```bash
cd script
python question_optimizer_agent.py
```

Input question in format: `题干;选项;答案;知识点;章节;难度`. The agent will optimize through multi-turn conversation.

### License

MIT License
