"""
RAG试题生成服务 - 从graphRAG.ipynb封装
"""
import os
import random
from math import exp
from collections import defaultdict, Counter

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
from langchain_community.graphs import Neo4jGraph
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

# ============ 配置 ============
# NEO4J_URL = "bolt://localhost:7689"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "12345678"

from dotenv import load_dotenv
load_dotenv()
import os

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 向量库相关
VECTORSTORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vectorstore")


# 学生群体参数
stu_map = [
    {'handle': 0.5, 'calculation': 0.7, 'cognitive_levels': 3},
    {'handle': 0.2, 'calculation': 0.5, 'cognitive_levels': 2},
    {'handle': 0.7, 'calculation': 0.5, 'cognitive_levels': 4}
]

# 小节名称映射
block_names = {
    0: '流水地貌',
    1: '风成地貌',
    2: '喀斯特、海岸和冰川地貌'
}

# ============ 题库特征模型 ============
feature_map = {
    '流水地貌': {
        'k_points': ["地貌", "流水地貌", "流水侵蚀地貌", "流水堆积地貌", "峡谷", "河漫滩",
                    "河流阶地", "曲流", "牛轭湖", "冲积扇", "洪积扇", "冲积平原", "三角洲",
                    "江心洲", "滑坡", "泥石流", "流水", "山洪", "虎跳峡", "雅鲁藏布大峡谷",
                    "嘉陵江", "崇明岛", "橘子洲", "黄河三角洲", "尼罗河三角洲", "华北平原",
                    "东北平原", "长江中下游平原"],
        'difficulty': [33/43, 8/43, 2/43],
        'q_type': [1],
        'calculation': 0.0,
        'cognitive_levels': [5/43, 30/43, 0/43, 8/43, 0/43, 0/43]
    },
    '风成地貌': {
        'k_points': ["风蚀地貌", "风积地貌", "风蚀蘑菇", "风蚀壁龛", "风蚀柱", "雅丹地貌",
                    "沙丘", "新月形沙丘", "灌丛沙丘", "风力"],
        'difficulty': [13/23, 10/23, 0/23],
        'q_type': [1],
        'calculation': 0.0,
        'cognitive_levels': [12/23, 7/23, 1/23, 3/23, 0/23, 0/23]
    },
    '喀斯特、海岸和冰川地貌': {
        'k_points': ["喀斯特地貌", "海岸地貌", "冰川地貌", "海蚀地貌", "海积地貌",
                    "海蚀崖", "海蚀平台", "海蚀柱", "海滩", "岬角", "冰斗", "冰川槽谷",
                    "角峰", "刃脊", "峡湾", "波浪", "冰川", "云南石林", "广西桂林",
                    "重庆奉节小寨天坑", "贵州荔波", "四川黄龙", "湖南张家界黄龙洞",
                    "澳大利亚坎贝尔港国家公园", "挪威西峡湾"],
        'difficulty': [17/37, 1/37, 19/37],
        'q_type': [1],
        'calculation': 0.0,
        'cognitive_levels': [15/37, 4/37, 2/37, 16/37, 0/37, 0/37]
    }
}

# 常量列表
difficulty_list = ['简单', '中等', '困难']
q_type_list = ['选择题']
calculation_list = ['不涉及', '涉及']
cognitive_levels = ['记忆', '理解', '应用', '分析', '评价', '创造']


# ============ 初始化全局变量 ============
graph = None
llm = None
answer_propmt = None
vectorstore = None


def load_vectorstore():
    """加载向量库"""
    global vectorstore
    if vectorstore is None:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def retrieve_vector(vectorstore, query, k=3):
    """向量检索：返回相似文档"""
    docs = vectorstore.similarity_search(query, k=k)
    return '\n\n'.join([d.page_content for d in docs])


def init_services():
    """初始化Neo4j和LLM服务"""
    global graph, llm, answer_propmt

    if graph is None:
        load_dotenv()
        graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD, database="4bc8c2ea")


    if llm is None:
        llm = init_chat_model(model='deepseek:deepseek-chat', temperature=1.3)

    if answer_propmt is None:
        answer_propmt = PromptTemplate.from_template(
            '''
            你是一名高中地理资深命题教师。

            【已知条件】
            - 知识上下文：
            {context}
            - 目标知识点：{knowledge_point}
            - 难度等级：{difficulty}
            - 题型：{q_type}
            - 是否涉及运算：{calculation}
            - 认知层级（如：记忆 / 理解 / 应用 / 分析 / 评价 / 创造）：
            {cognitive_level}

            【出题要求】
            1. 题目必须以「目标知识点」为核心
            2. 可合理使用上下文中的相关知识点作为支撑，但只能起辅助作用
            3. 严格匹配给定的题型、难度、是否运算、认知层级
            4. 题干表述清晰，不引入无关信息。纯文本题，不要用到图片。
            5. 设问方式灵活，所问的可能是下一句或者选择一个陈述句（这两种情况最多），也可能是挖空，也可能是从下列选择或排序等
            5. 有概率用真实地理景观为例出题，但不要只用上下文中的，可以用其他的，比如"乞拉朋齐……据此请分析……的成因主要是"
            5. 只生成一道题，不要解释、不输出思路
            6. 要符合高中地理考题风格

            严格按照以下格式（包括换行也要一致）输出，不要含有其他内容如前后缀等，最前面不要有空格
            【输出格式】：
            【{difficulty}题 | {cognitive_level}题 | 运算题（如果涉及运算的话）】前两个xx分别填难度、认知层级
            【题目】
            （另起一行）题目内容
            （如是选择题）【选项】
            【参考答案】
            （另起一行）（如果是选择题，只输出选项一个字母）
            '''
        )

    return graph, llm


# ============ 核心函数 ============
def retrieve_1hop_subgraph(graph, keyword):
    """图检索：返回子图"""
    query = """
    MATCH (n {id: $kw})
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN
        COLLECT(DISTINCT n) + COLLECT(DISTINCT m) AS nodes,
        COLLECT(DISTINCT r) AS relationships
    """
    res = graph.query(query, {"kw": keyword})
    if not res or not res[0]["nodes"]:
        return {"nodes": [], "relationships": []}
    return {"nodes": res[0]["nodes"], "relationships": res[0]["relationships"]}


def get_knowledge_context(subgraph):
    """从子图构建知识上下文"""
    sentences = set()
    for rel in subgraph["relationships"]:
        source = rel[0]['id']
        target = rel[2]['id']
        rtype = rel[1]
        sentence = f"{source}{rtype}{target}。"
        sentences.add(sentence)
    return "\n".join(sorted(sentences))


def rag(knowledge_point, difficulty, q_type, calculation, cognitive_level):
    """生成函数：检索+生成"""
    graph_results = retrieve_1hop_subgraph(graph, knowledge_point)
    context = get_knowledge_context(graph_results)

    response = llm.invoke(
        answer_propmt.format(
            context=context,
            knowledge_point=knowledge_point,
            difficulty=difficulty,
            q_type=q_type,
            calculation=calculation,
            cognitive_level=cognitive_level
        )
    )

    return graph_results, context, response.content


def rag_vector(knowledge_point, difficulty, q_type, calculation, cognitive_level):
    """向量检索生成函数"""
    vs = load_vectorstore()
    context = retrieve_vector(vs, knowledge_point, k=3)

    response = llm.invoke(
        answer_propmt.format(
            context=context,
            knowledge_point=knowledge_point,
            difficulty=difficulty,
            q_type=q_type,
            calculation=calculation,
            cognitive_level=cognitive_level
        )
    )

    return {"nodes": []}, context, response.content


def rag_no_retrieval(knowledge_point, difficulty, q_type, calculation, cognitive_level):
    """无检索生成函数（直接生成，不注入知识上下文）"""
    response = llm.invoke(
        answer_propmt.format(
            context="",
            knowledge_point=knowledge_point,
            difficulty=difficulty,
            q_type=q_type,
            calculation=calculation,
            cognitive_level=cognitive_level
        )
    )

    return {"nodes": []}, "", response.content


def merge_feature(block, stu_group):
    """融合题库特征与学生背景"""
    true_feature = feature_map[block]
    stu_bg = stu_map[stu_group]

    # 难度
    t = 0.5
    difficulty_spot = [0.2, 0.5, 0.85]
    difficulty = true_feature['difficulty']
    difficulty_p = []
    sum_p = 0
    for i in range(3):
        distance = abs(difficulty[i] - difficulty_spot[i])
        w_i = exp(-distance / t)
        p = difficulty[i] * w_i
        difficulty_p.append(p)
        sum_p += p
    for i in range(3):
        difficulty_p[i] = difficulty_p[i] / sum_p

    # 运算概率
    calculation_prob = true_feature['calculation'] * (1.5 - stu_bg['calculation'])

    # 认知层级
    decay1 = [0.3, 0.4, 0.5, 0.7, 0.8, 1]
    decay2 = [1, 0.5, 0.25, 0.1, 0.1, 0.1]
    cognitive_levels = true_feature['cognitive_levels']
    c = stu_bg['cognitive_levels']
    cognitive_p = [0 for _ in range(6)]
    for i in range(c + 1):
        cognitive_p[i] = cognitive_levels[i] * decay1[6 - c + i - 1]
    if c < 5:
        for i in range(c + 1, 6):
            cognitive_p[i] = cognitive_levels[i] * decay2[i - c]
    sum_p = sum(cognitive_p)
    for i in range(6):
        cognitive_p[i] = cognitive_p[i] / sum_p

    return difficulty_p, calculation_prob, cognitive_p


def choose_feature(difficulty_p, calculation, cognitive_p):
    """根据概率采样具体参数"""
    cur_difficulty = random.choices([0, 1, 2], weights=difficulty_p, k=1)[0]
    cur_difficulty = difficulty_list[cur_difficulty]

    cur_calculation = random.random() < calculation
    cur_calculation = calculation_list[int(cur_calculation)]

    cur_cognitive_level = random.choices(range(6), weights=cognitive_p, k=1)[0]
    cur_cognitive_level = cognitive_levels[cur_cognitive_level]

    return cur_difficulty, cur_calculation, cur_cognitive_level


def get_questions(block_idx, stu_group, knowledge_point=None):
    """
    生成试题的主函数

    参数:
        block_idx: int, 0=流水地貌, 1=风成地貌, 2=喀斯特海岸冰川地貌
        stu_group: int, 0=A类学生, 1=B类学生
        knowledge_point: str, 可选，指定知识点

    返回:
        dict: {
            "question": str, 生成的题目,
            "features": list, [知识点, 难度, 题型, 是否运算, 认知层级],
            "context": str, 知识上下文
        }
    """
    # 初始化服务
    init_services()

    # 获取小节名称
    block = block_names.get(block_idx, '流水地貌')

    # 抽取生成要求
    difficulty_p, calculation, cognitive_p = merge_feature(block, stu_group)
    cur_difficulty, cur_calculation, cur_cognitive_level = choose_feature(difficulty_p, calculation, cognitive_p)

    # 抽取知识点
    all_kps = feature_map[block]['k_points']
    if not knowledge_point or knowledge_point not in all_kps:
        knowledge_point = random.choices(all_kps, k=1)[0]

    # 生成
    graph_results, context, answer = rag(
        knowledge_point=knowledge_point,
        difficulty=cur_difficulty,
        q_type=q_type_list[0],
        calculation=cur_calculation,
        cognitive_level=cur_cognitive_level
    )

    features = [knowledge_point, cur_difficulty, q_type_list[0], cur_calculation, cur_cognitive_level]

    return {
        "question": answer,
        "features": features,
        "context": context
    }


def get_questions_with_vector(block_idx, stu_group, knowledge_point=None):
    """
    使用向量检索生成试题的主函数

    参数:
        block_idx: int, 0=流水地貌, 1=风成地貌, 2=喀斯特海岸冰川地貌
        stu_group: int, 0=A类学生, 1=B类学生
        knowledge_point: str, 可选，指定知识点

    返回:
        dict: {
            "question": str, 生成的题目,
            "features": list, [知识点, 难度, 题型, 是否运算, 认知层级],
            "context": str, 知识上下文
        }
    """
    # 初始化服务
    init_services()

    # 获取小节名称
    block = block_names.get(block_idx, '流水地貌')

    # 抽取生成要求
    difficulty_p, calculation, cognitive_p = merge_feature(block, stu_group)
    cur_difficulty, cur_calculation, cur_cognitive_level = choose_feature(difficulty_p, calculation, cognitive_p)

    # 抽取知识点
    all_kps = feature_map[block]['k_points']
    if not knowledge_point or knowledge_point not in all_kps:
        knowledge_point = random.choices(all_kps, k=1)[0]

    # 生成（使用向量检索）
    graph_results, context, answer = rag_vector(
        knowledge_point=knowledge_point,
        difficulty=cur_difficulty,
        q_type=q_type_list[0],
        calculation=cur_calculation,
        cognitive_level=cur_cognitive_level
    )

    features = [knowledge_point, cur_difficulty, q_type_list[0], cur_calculation, cur_cognitive_level]

    return {
        "question": answer,
        "features": features,
        "context": context
    }


def get_questions_no_retrieval(block_idx, stu_group, knowledge_point=None):
    """
    无检索生成试题的主函数

    参数:
        block_idx: int, 0=流水地貌, 1=风成地貌, 2=喀斯特海岸冰川地貌
        stu_group: int, 0=A类学生, 1=B类学生
        knowledge_point: str, 可选，指定知识点

    返回:
        dict: {
            "question": str, 生成的题目,
            "features": list, [知识点, 难度, 题型, 是否运算, 认知层级],
            "context": str, 知识上下文
        }
    """
    # 初始化服务
    init_services()

    # 获取小节名称
    block = block_names.get(block_idx, '流水地貌')

    # 抽取生成要求
    difficulty_p, calculation, cognitive_p = merge_feature(block, stu_group)
    cur_difficulty, cur_calculation, cur_cognitive_level = choose_feature(difficulty_p, calculation, cognitive_p)

    # 抽取知识点
    all_kps = feature_map[block]['k_points']
    if not knowledge_point or knowledge_point not in all_kps:
        knowledge_point = random.choices(all_kps, k=1)[0]

    # 生成（无检索）
    graph_results, context, answer = rag_no_retrieval(
        knowledge_point=knowledge_point,
        difficulty=cur_difficulty,
        q_type=q_type_list[0],
        calculation=cur_calculation,
        cognitive_level=cur_cognitive_level
    )

    features = [knowledge_point, cur_difficulty, q_type_list[0], cur_calculation, cur_cognitive_level]

    return {
        "question": answer,
        "features": features,
        "context": context
    }


def check_health():
    """检查服务状态"""
    try:
        init_services()
        # 简单测试图数据库连接
        graph.query("RETURN 1")
        return {"status": "ok", "message": "服务正常"}
    except Exception as e:
        return {"status": "error", "message": str(e)}