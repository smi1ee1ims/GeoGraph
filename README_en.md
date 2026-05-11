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

```
backend/
├── main.py              # FastAPI entry point
├── rag_service.py      # Core RAG service
├── build_vectorstore.py # Vector store builder
├── index.html          # Frontend page
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
└── data/
    ├── cleaned_湘教版地理必修一Chap2.md  # Textbook content
    └── vectorstore/    # FAISS vector store
```

### 【5】Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables (.env)
NEO4J_URL=bolt://localhost:7689
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# 3. Build vector store (first run only)
python build_vectorstore.py

# 4. Start the service
python main.py
```

Access `http://localhost` to use the frontend.

### 【6】Core Scripts (without frontend)

#### graphRAG.ipynb - Question Generation
Open in Jupyter notebook and run cells directly. Supports three modes:
- **Graph Retrieval**: Uses Neo4j knowledge graph
- **Vector Retrieval**: Uses FAISS vector store
- **No Retrieval**: Direct LLM generation

#### eval.ipynb - Evaluators 1-3
Three evaluation metrics:
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
