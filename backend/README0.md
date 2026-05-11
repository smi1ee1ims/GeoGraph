# GeoQuestionGen - 个性化地理试题生成系统

[中文](README.md) | [English](README_en.md)

### 项目简介

SAMEDNA 是一个基于检索增强生成（RAG）的个性化高中地理试题生成系统，支持**图检索**、**向量检索**和**无检索**三种生成方式，可根据学生群体特征自适应调整试题难度和认知层级。

**适用教材**：湘教版地理必修一 第二章「地球表面形态」

### 核心功能

- **图检索生成**：基于 Neo4j 知识图谱进行关系路径检索
- **向量检索生成**：基于 FAISS 语义向量库进行相似文档检索
- **无检索生成**：直接由 LLM 根据知识点生成
- **自适应出题**：融合题库特征模型与学生背景，动态调整：
  - 难度等级（简单 / 中等 / 困难）
  - 认知层级（记忆 → 理解 → 应用 → 分析 → 评价 → 创造）
  - 运算概率

### 章节覆盖

| 编号 | 章节 |
|------|------|
| 0 | 2.1 流水地貌 |
| 1 | 2.2 风成地貌 |
| 2 | 2.3 喀斯特、海岸和冰川地貌 |

### 技术架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   前端页面   │ ──▶ │   FastAPI    │ ──▶ │  LangChain  │
│  (HTML/jQuery) │     │   后端服务   │     │    LLM 调用  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                    ┌──────┴──────┐     ┌──────┴──────┐
                    │   Neo4j     │     │   DeepSeek  │
                    │  知识图谱   │     │    LLM      │
                    └─────────────┘     └─────────────┘
```

### 快速部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量 (.env)
NEO4J_URL=bolt://localhost:7689
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# 3. 构建向量库（首次运行）
python build_vectorstore.py

# 4. 启动服务
python main.py
```

访问 `http://localhost` 即可使用前端页面。

### API 接口

**生成试题**
```http
POST /api/generate
Content-Type: application/json

{
  "block": 0,           // 0=流水地貌, 1=风成地貌, 2=喀斯特海岸冰川地貌
  "stu_group": 0,       // 0=A类学生, 1=B类学生
  "knowledge_point": null,  // 可选，指定知识点
  "method": 0           // 0=图检索, 1=向量检索, 2=无检索
}
```

**健康检查**
```http
GET /health
```

### 项目结构

```
backend/
├── main.py              # FastAPI 入口
├── rag_service.py      # 核心 RAG 服务
├── build_vectorstore.py # 向量库构建脚本
├── index.html          # 前端页面
├── requirements.txt    # 依赖列表
├── .env.example        # 环境变量模板
└── data/
    ├── cleaned_湘教版地理必修一Chap2.md  # 教材文本
    └── vectorstore/    # FAISS 向量库
```

### License

MIT License
