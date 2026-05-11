# GeoAgent - 你的个性化智能地理出题助手

[中文](./README.md) | [English](./README_en.md)

项目网页试用地址：http://47.111.172.69/

欢迎您的宝贵建议和交流：1345119185@qq.com（QQ号同）

### ⏫2026.5.11更新
- 🪐支持和agent交流打磨试题
- 🪐开放了专属评估器、优化器
- 🪐重构UI，操作更便捷

### 【1】项目简介
嗨！GeoAgent是一个基于多路RAG的**个性化**地理试题助手，两大亮点：

1.**专有知识库**：匹配支持**图检索**、**向量检索**和**无检索**三种生成方式

2.**个性化定制**：可根据学生群体特征自适应调整试题难度和认知层级。
<!-- <img width="1379" height="510" alt="image" src="https://github.com/user-attachments/assets/1e7fcf35-66eb-4093-a050-252afc5fca02" /> -->
<img width="1893" height="884" alt="image" src="https://github.com/user-attachments/assets/2e013195-2175-43a7-ab4a-8feca727e155" />

接入deepseek-chat-V3.2 & qwen-max供选择

目前已开放第二章测试功能，更多章节敬请期待。

### 【2】核心功能

- **图检索生成**：基于 Neo4j 知识图谱进行关系路径检索
- **向量检索生成**：基于 FAISS 语义向量库进行相似文档检索
- **无检索生成**：直接由 LLM 根据知识点生成
- **自适应出题**：融合题库特征模型与学生背景，动态调整。
  - 难度等级（简单 / 中等 / 困难）
  - 认知层级（记忆 → 理解 → 应用 → 分析 → 评价 → 创造）
  - 运算概率

### 【3】技术栈
- 前端:HTML;jQuery
- 后端:FastAPI
- agent链路:LangChain
- 数据库：Neo4j;AuraDB;VectorStore
- LLMapi:deepseek-chat-v3.2;qwen-max

### 【4】项目结构

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
### 【5】快速部署

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

### License

MIT License
