"""
Question Optimization Agent
===========================
试题优化 Agent - 集成评估函数，支持多 LLM 提供商

使用方法：
    python question_optimizer_agent.py --demo    # 演示模式（真实API调用）
    python question_optimizer_agent.py --optimize # 真实优化模式
"""

import os

# 自动加载 .env 文件
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    with open(_env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

import re
from dataclasses import dataclass, field
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# =============================================================================
# LLM 提供商配置
# =============================================================================

LLM_PROVIDERS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
        "default_model": "claude-sonnet-4-20250514"
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat"
    },
    "minimax": {
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "abab6.5s-chat"
    }
}


def get_llm_client(provider: str = "deepseek", api_key: str = None):
    """获取 LLM 客户端"""
    config = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["deepseek"])

    if not api_key:
        api_key = os.environ.get(config["env_key"])

    if not api_key:
        raise ValueError(f"需要 API key，请设置环境变量 {config['env_key']}")

    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    else:
        # OpenAI-compatible (deepseek, minimax)
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=config["base_url"])


def llm_chat(client, messages: list, model: str = None) -> str:
    """统一的 chat 接口"""
    if hasattr(client, 'messages'):
        # Anthropic
        response = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages
        )
        return response.content[0].text
    else:
        # OpenAI-compatible
        response = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=messages
        )
        return response.choices[0].message.content


# =============================================================================
# 评估函数（从 eval.ipynb 集成）
# =============================================================================

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("警告: sentence-transformers 未安装，难度评估不可用")


def compute_knowledge_relevance(full_question: str, knowledge_context: str) -> float:
    """计算知识相关性（余弦相似度）"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return 0.5

    if not knowledge_context or knowledge_context is None:
        return 0.0

    import pandas as pd
    if pd.isna(knowledge_context):
        return 0.0

    if not hasattr(compute_knowledge_relevance, '_embedder'):
        compute_knowledge_relevance._embedder = SentenceTransformer('all-MiniLM-L6-v2')

    embedder = compute_knowledge_relevance._embedder

    sentences = [s.strip() for s in knowledge_context.split('。') if s.strip()]
    if not sentences:
        return 0.0

    import numpy as np
    q_emb = embedder.encode(full_question, convert_to_tensor=True)
    s_embs = embedder.encode(sentences, convert_to_tensor=True)

    from sentence_transformers import util
    cos_scores = util.cos_sim(q_emb, s_embs)[0].cpu().numpy()

    k = min(3, len(sentences))
    top_k_idx = np.argsort(cos_scores)[-k:]
    return float(cos_scores[top_k_idx].mean())


def compute_explainability_single(question: str, options: str, llm_client, n: int = 5) -> float:
    """通过 LLM 多次回答评估答案一致性"""
    system_prompt = '你是一名资深地理教师，请回答以下选择题。只输出选项字母（A、B、C、D）中的一个。'
    user_prompt = f"题目：{question}\n{options}\n你的答案："

    answers = []
    for _ in range(n):
        try:
            response = llm_chat(llm_client, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            match = re.search(r'[A-D]', response)
            answers.append(match.group(0) if match else 'ERROR')
        except Exception as e:
            print(f"explainability error: {e}")
            answers.append('ERROR')

    counter = Counter(answers)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / n if n > 0 else 0


def eval_difficulty_accuracy_single(
    question: str, options: str, target_difficulty: str,
    section: str, bank_xlsx: str
) -> float:
    """评估难度准确性（与题库相似度）"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return 0.5

    if not os.path.exists(bank_xlsx):
        return 0.5

    import pandas as pd
    import numpy as np

    if not hasattr(eval_difficulty_accuracy_single, '_model'):
        eval_difficulty_accuracy_single._model = SentenceTransformer('all-MiniLM-L6-v2')
    if not hasattr(eval_difficulty_accuracy_single, '_bank_cache') or \
       eval_difficulty_accuracy_single._bank_cache.get('path') != bank_xlsx:
        df = pd.read_excel(bank_xlsx, dtype=str)
        bank_items = []
        for idx, row in df.iterrows():
            sec = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ''
            diff = str(row.iloc[9]) if pd.notna(row.iloc[9]) else ''
            parts = [str(row.iloc[i]) if pd.notna(row.iloc[i]) else '' for i in range(2, 8)]
            full_text = ' '.join([p for p in parts if p.strip()])
            if full_text.strip():
                emb = eval_difficulty_accuracy_single._model.encode(full_text, normalize_embeddings=True)
                bank_items.append((sec, diff, emb))
        eval_difficulty_accuracy_single._bank_cache = {'path': bank_xlsx, 'items': bank_items}

    bank_items = eval_difficulty_accuracy_single._bank_cache['items']
    full_q = question + ' ' + options
    q_emb = eval_difficulty_accuracy_single._model.encode(full_q, normalize_embeddings=True)

    candidates = [item for item in bank_items if item[0] == section and item[1] == target_difficulty]
    if not candidates:
        return 0.0

    bank_embs = np.array([item[2] for item in candidates])
    return float(np.max(np.dot(bank_embs, q_emb)))


# =============================================================================
# 配置常量
# =============================================================================

THRESHOLDS = {
    "relevance": 0.7,
    "explainability": 0.9,
    "difficulty": 0.5
}
MAX_ITERATIONS = 3

# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class QuestionData:
    question: str
    options: str
    answer: str
    knowledge_context: str
    features: dict = field(default_factory=dict)
    target_difficulty: str = "中等"


@dataclass
class EvaluationResult:
    metric: str
    score: float
    passed: bool
    details: str
    improvement_hints: list


# =============================================================================
# 评估器
# =============================================================================

class EvaluatorSubagent:
    def __init__(self, metric: str, evaluate_func: Callable, threshold: float):
        self.metric = metric
        self.evaluate_func = evaluate_func
        self.threshold = threshold

    def evaluate(self, question_data: QuestionData, llm_client=None) -> EvaluationResult:
        try:
            if self.metric == "relevance":
                score = self.evaluate_func(question_data.question, question_data.knowledge_context)
            elif self.metric == "explainability":
                if not llm_client:
                    return EvaluationResult(self.metric, 0.0, False, "No LLM client", ["需要 LLM 客户端"])
                score = self.evaluate_func(question_data.question, question_data.options, llm_client)
            elif self.metric == "difficulty":
                score = self.evaluate_func(
                    question_data.question, question_data.options,
                    question_data.target_difficulty,
                    question_data.features.get("section", ""),
                    question_data.features.get("bank_path", "")
                )
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            passed = score >= self.threshold
            hints = [] if passed else self._get_hints(self.metric)

            return EvaluationResult(self.metric, score, passed,
                f"{self.metric}: {score:.3f} {'>=' if passed else '<'} {self.threshold}",
                hints)
        except Exception as e:
            return EvaluationResult(self.metric, 0.0, False, f"Error: {e}", [str(e)])

    def _get_hints(self, metric: str) -> list:
        return {
            "relevance": ["增加与知识上下文的关联", "使用更多上下文中的术语"],
            "explainability": ["减少歧义", "增强答案唯一性"],
            "difficulty": ["调整复杂度", "匹配目标难度"]
        }.get(metric, [])


def parallel_evaluate(evaluators, question_data, llm_client=None):
    results = {}
    with ThreadPoolExecutor(max_workers=len(evaluators)) as executor:
        futures = {executor.submit(e.evaluate, question_data, llm_client): e.metric for e in evaluators}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


# =============================================================================
# 优化器
# =============================================================================

class OptimizerSubagent:
    def __init__(self, focus_metric: str, llm_client):
        self.focus = focus_metric
        self.llm = llm_client

    def optimize(self, question: QuestionData, eval_result: EvaluationResult) -> str:
        knowledge_point = question.features.get("knowledge_point", "")
        kp_constraint = f"\n核心知识点：{knowledge_point}" if knowledge_point else ""

        prompt = f"""你是一名高中地理命题教师。请根据评估反馈优化试题。

【原始试题】
题目：{question.question}
选项：{question.options}
答案：{question.answer}
知识上下文：{question.knowledge_context}{kp_constraint}

【评估结果】
- 指标：{eval_result.metric}
- 分数：{eval_result.score:.3f}
- 通过：{'是' if eval_result.passed else '否'}
- 改进建议：{', '.join(eval_result.improvement_hints) or '无'}

【要求】
1. 必须围绕核心知识点{knowledge_point if knowledge_point else ""}出题
2. 保留正确部分
3. 针对未通过指标优化
4. 直接输出优化后的试题（题目+选项+答案）
5. 不要解释
"""

        try:
            return llm_chat(self.llm, [{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"Optimization error: {e}")
            return question.question


class MultiObjectiveOptimizer:
    def __init__(self, llm_client):
        self.llm = llm_client

    def optimize(self, question: QuestionData, failing_metrics: dict) -> str:
        metrics_info = "\n".join([
            f"- {m}: 分数={r.score:.3f}, 建议={', '.join(r.improvement_hints)}"
            for m, r in failing_metrics.items()
        ])
        knowledge_point = question.features.get("knowledge_point", "")
        kp_constraint = f"\n核心知识点：{knowledge_point}" if knowledge_point else ""

        prompt = f"""你是一名高中地理命题教师。当多个指标失败时需要平衡优化。

【原始试题】
题目：{question.question}
选项：{question.options}
答案：{question.answer}
知识上下文：{question.knowledge_context}{kp_constraint}

【失败指标】
{metrics_info}

【要求】
1. 必须围绕核心知识点{knowledge_point if knowledge_point else ""}出题
2. 同时提升所有失败指标
3. 保持正确性
4. 直接输出优化后的试题（题目+选项+答案）
5. 不要解释
"""
        try:
            return llm_chat(self.llm, [{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"Multi-optimization error: {e}")
            return question.question


# =============================================================================
# 主类
# =============================================================================

class QuestionOptimizer:
    def __init__(self, provider: str = "deepseek", api_key: str = None):
        self.llm = get_llm_client(provider, api_key)

        self.evaluators = [
            EvaluatorSubagent("relevance", compute_knowledge_relevance, THRESHOLDS["relevance"]),
            EvaluatorSubagent("explainability", compute_explainability_single, THRESHOLDS["explainability"]),
            EvaluatorSubagent("difficulty", eval_difficulty_accuracy_single, THRESHOLDS["difficulty"]),
        ]

        self.single_optimizers = {
            "relevance": OptimizerSubagent("relevance", self.llm),
            "explainability": OptimizerSubagent("explainability", self.llm),
            "difficulty": OptimizerSubagent("difficulty", self.llm),
        }
        self.multi_optimizer = MultiObjectiveOptimizer(self.llm)

    def optimize(self, question_data: QuestionData):
        """运行优化流程"""
        question = question_data
        best_question = question_data
        best_avg_score = 0.0
        history = []

        for i in range(MAX_ITERATIONS):
            print(f"\n=== 迭代 {i+1}/{MAX_ITERATIONS} ===")

            results = parallel_evaluate(self.evaluators, question, self.llm)

            scores = {}
            for metric, result in results.items():
                scores[metric] = result.score
                print(f"  {metric}: {result.score:.3f} {'✅' if result.passed else '❌'}")

            avg = sum(scores.values()) / len(scores)
            print(f"  平均分: {avg:.3f}")

            history.append({"iteration": i+1, "scores": scores.copy(), "avg": avg})

            if avg > best_avg_score:
                best_question = question
                best_avg_score = avg
                print(f"  📌 新最佳: {best_avg_score:.3f}")

            if all(r.passed for r in results.values()):
                print("  🎉 全部通过!")
                break

            failing = {k: v for k, v in results.items() if not v.passed}
            print(f"  ⚠️ 未通过: {list(failing.keys())}")

            if len(failing) >= 2:
                print("  → 多目标优化")
                new_q = self.multi_optimizer.optimize(question, failing)
            else:
                metric = list(failing.keys())[0]
                print(f"  → 单目标优化 ({metric})")
                new_q = self.single_optimizers[metric].optimize(question, failing[metric])

            question = QuestionData(
                question=new_q,
                options=question.options,
                answer=question.answer,
                knowledge_context=question.knowledge_context,
                features=question.features,
                target_difficulty=question.target_difficulty
            )

        return {
            "final_question": question,
            "best_question": best_question,
            "best_avg_score": best_avg_score,
            "history": history
        }


# =============================================================================
# 演示和主入口
# =============================================================================

def run_demo():
    """演示模式"""
    print("=" * 60)
    print("试题优化 Agent - 演示模式")
    print("=" * 60)

    try:
        optimizer = QuestionOptimizer(provider="deepseek")
        print("✅ 优化器创建成功!")
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        print(f"请设置 DEEPSEEK_API_KEY 环境变量")
        return

    demo_question = QuestionData(
        question="虎跳峡位于中国西南地区，是世界上最深的峡谷之一，以水流湍急、峡谷幽深著称。峡谷两岸的岩层陡峭，谷底狭窄，江心有一巨石（虎跳石）矗立。该地貌的形成与下列哪一因素关系最为密切？",
        options="A. 冰川的刨蚀作用\nB. 地壳的持续抬升\nC. 流水的溯源侵蚀\nD. 风力的长期吹蚀",
        answer="C",
        knowledge_context="虎跳峡位于中国西南地区。流水侵蚀地貌形成于虎跳峡。虎跳峡是典型流水侵蚀地貌。",
        features={"section": "流水地貌", "bank_path": "./data/题库 _已标注.xlsx"},
        target_difficulty="中等"
    )

    print(f"\n📌 测试数据:")
    print(f"   题目: {demo_question.question[:60]}...")
    print(f"   知识上下文: {demo_question.knowledge_context[:40]}...")

    print("\n🚀 开始优化...")
    result = optimizer.optimize(demo_question)

    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    print(f"迭代次数: {len(result['history'])}")
    print(f"最佳平均分: {result['best_avg_score']:.3f}")
    print(f"\n最佳试题:")
    print(f"  {result['best_question'].question}")
    print(f"  {result['best_question'].options}")

    print("\n评估历史:")
    for h in result['history']:
        print(f"  迭代{h['iteration']}: {h['avg']:.3f}")


# =============================================================================
# 工具定义（用于 Tool Router）
# =============================================================================

TOOLS = [
    {
        "name": "submit_question",
        "description": "提交一道新试题进行评估和优化。参数：题干;选项;答案;知识上下文;章节;难度;知识点（知识点可选）",
        "parameters": {
            "type": "object",
            "properties": {
                "question_data": {"type": "string", "description": "试题数据，格式：题干;选项;答案;知识上下文;章节;难度;知识点"}
            },
            "required": ["question_data"]
        }
    },
    {
        "name": "optimize_question",
        "description": "对当前试题进行优化评估。如果有未通过的指标，会自动优化试题。",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "modify_question",
        "description": "根据用户要求修改试题。例如：'把题目难度改成困难'、'丰富题干描述'、'换一个知识点'",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "用户的修改指令"}
            },
            "required": ["instruction"]
        }
    },
    {
        "name": "show_status",
        "description": "显示当前试题的状态，包括评估结果和历史",
            "parameters": {
                "type": "object",
                "properties": {}
            }
    },
    {
        "name": "chat",
        "description": "回答用户的问题，和用户聊天。可以回答关于试题、评估、优化流程等问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "用户的消息"}
            },
            "required": ["message"]
        }
    }
]


def build_tools_prompt():
    """构建工具描述用于 LLM"""
    tools_desc = []
    for t in TOOLS:
        params = t.get("parameters", {}).get("properties", {})
        param_str = ", ".join([f"{k}: {v.get('description', k)}" for k, v in params.items()])
        tools_desc.append(f"- {t['name']}: {t['description']} (参数: {param_str})")
    return "\n".join(tools_desc)


class ToolRouter:
    """根据用户意图路由到对应工具"""

    def __init__(self, optimizer, current_question):
        self.optimizer = optimizer
        self.current_question = current_question

    def route(self, user_message: str) -> dict:
        """分析用户消息，决定调用哪个工具"""
        # 构建 prompt 让 LLM 决定
        has_question = self.current_question is not None

        prompt = f"""用户说："{user_message}"

当前状态：
- 已有试题：{'是' if has_question else '否'}
{f'- 试题：{self.current_question.question[:50]}...' if has_question else ''}
{f'- 知识点：{self.current_question.features.get("knowledge_point", "")}' if has_question else ''}

请判断用户意图，选择最合适的工具。

可用工具：
{build_tools_prompt()}

直接输出 JSON 格式：
{{"tool": "工具名", "args": {{"参数名": "参数值"}}}} 或者 {{"tool": "chat", "args": {{"message": "..."}}}}
不要输出其他内容。
"""

        try:
            response = llm_chat(self.optimizer.llm, [
                {"role": "user", "content": prompt}
            ])

            # 解析 JSON 响应
            import json
            import re

            # 尝试提取 JSON
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                result = json.loads(match.group())
                return result
        except Exception as e:
            print(f"路由错误: {e}")

        # fallback 到 chat
        return {"tool": "chat", "args": {"message": user_message}}


def execute_tool(tool_name: str, args: dict, optimizer, current_question):
    """执行工具并返回结果"""
    result = {"success": False, "message": "", "data": None}

    if tool_name == "submit_question":
        # 解析试题数据
        question_data = args.get("question_data", "")
        parts = question_data.split(';')
        if len(parts) < 4:
            result["message"] = "需要至少：题干;选项;答案;知识上下文"
            return result, None

        question = parts[0].strip()
        options = parts[1].strip()
        answer_raw = parts[2].strip()

        # 解析答案
        answer = answer_raw.upper() if answer_raw in ['A', 'B', 'C', 'D'] else None
        if not answer:
            for letter in ['A', 'B', 'C', 'D']:
                opt_match = re.search(rf'{letter}\.\s*(.+?)(?=\s*[A-D]\.|$)', options)
                if opt_match and answer_raw in opt_match.group(1):
                    answer = letter
                    break
            if not answer:
                result["message"] = f"无法解析答案 '{answer_raw}'"
                return result, None

        knowledge_context = parts[3].strip()
        section = parts[4].strip() if len(parts) > 4 else "流水地貌"
        difficulty = parts[5].strip() if len(parts) > 5 else "中等"
        knowledge_point = parts[6].strip() if len(parts) > 6 else ""

        if difficulty not in ["简单", "中等", "困难"]:
            result["message"] = f"难度必须是 简单/中等/困难 之一"
            return result, None

        q = QuestionData(
            question=question,
            options=options,
            answer=answer,
            knowledge_context=knowledge_context,
            features={"section": section, "bank_path": "./data/题库 _已标注.xlsx", "knowledge_point": knowledge_point},
            target_difficulty=difficulty
        )

        # 运行优化
        print("\n📊 评估中...")
        opt_result = optimizer.optimize(q)

        result["success"] = True
        result["message"] = f"评估完成 ({len(opt_result['history'])} 次迭代)"
        result["data"] = {
            "question": q,
            "result": opt_result
        }
        return result, q

    elif tool_name == "optimize_question":
        if not current_question:
            result["message"] = "请先提交试题"
            return result, None

        print("\n📊 重新评估中...")
        opt_result = optimizer.optimize(current_question)

        result["success"] = True
        result["message"] = f"优化完成 ({len(opt_result['history'])} 次迭代)"
        result["data"] = {
            "result": opt_result
        }
        return result, current_question

    elif tool_name == "modify_question":
        if not current_question:
            result["message"] = "请先提交试题"
            return result, None

        instruction = args.get("instruction", "")
        print(f"\n✏️ 修改试题: {instruction}")

        # 让 LLM 修改试题
        kp = current_question.features.get("knowledge_point", "")
        prompt = f"""用户要求修改试题：
指令：{instruction}

当前试题：
题目：{current_question.question}
选项：{current_question.options}
答案：{current_question.answer}
知识点：{kp}

请根据指令修改试题，直接输出修改后的完整试题（题目+选项+答案），不要解释。
"""
        new_question = llm_chat(optimizer.llm, [{"role": "user", "content": prompt}])

        # 更新试题
        new_q = QuestionData(
            question=new_question,
            options=current_question.options,
            answer=current_question.answer,
            knowledge_context=current_question.knowledge_context,
            features=current_question.features,
            target_difficulty=current_question.target_difficulty
        )

        result["success"] = True
        result["message"] = f"已修改"
        result["data"] = {"new_question": new_q}
        return result, new_q

    elif tool_name == "show_status":
        if not current_question:
            result["message"] = "暂无试题"
            return result, None

        result["success"] = True

        # 构建状态信息
        lines = ["📊 当前试题状态："]
        lines.append(f"   试题：{current_question.question[:50]}...")
        lines.append(f"   答案：{current_question.answer}")
        lines.append(f"   难度：{current_question.target_difficulty}")
        if current_question.features.get("knowledge_point"):
            lines.append(f"   知识点：{current_question.features['knowledge_point']}")

        if hasattr(current_question, 'evaluation_history') and current_question.evaluation_history:
            lines.append(f"\n📈 评估历史（共 {len(current_question.evaluation_history)} 次迭代）：")
            for h in current_question.evaluation_history:
                scores = ", ".join([f"{k}={v:.3f}" for k, v in h['scores'].items()])
                lines.append(f"   迭代{h['iteration']}: avg={h['avg']:.3f} [{scores}]")

            if hasattr(current_question, 'best_question') and current_question.best_question:
                bq = current_question.best_question
                lines.append(f"\n🏆 最佳试题：")
                lines.append(f"   {bq.question[:50]}...")
                lines.append(f"   答案: {bq.answer}")
        else:
            lines.append("\n⚠️ 暂无评估历史，请提交试题进行评估")

        result["message"] = "\n".join(lines)
        result["data"] = {"question": current_question}
        return result, current_question

    elif tool_name == "chat":
        msg = args.get("message", "")
        has_question = current_question is not None

        response = llm_chat(optimizer.llm, [
            {"role": "system", "content": f"""你是一个试题优化助手。

当前状态：
- 已有试题：{'是' if has_question else '否'}
{f'- 试题：{current_question.question[:80]}...' if has_question else ''}
{f'- 选项：{current_question.options[:80]}...' if has_question else ''}
{f'- 答案：{current_question.answer}' if has_question else ''}
{f'- 知识点：{current_question.features.get("knowledge_point", "")}' if has_question else ''}
{f'- 难度：{current_question.target_difficulty}' if has_question else ''}

你可以：
- 回答用户关于试题的问题
- 解释评估结果
- 提供改进建议
- 响应用户的修改要求

如果用户想提交试题或优化，请使用对应的工具。"""},
            {"role": "user", "content": msg}
        ])

        result["success"] = True
        result["message"] = response
        return result, current_question

    else:
        result["message"] = f"未知工具: {tool_name}"
        return result, current_question


# =============================================================================
# 主入口 - 对话循环
# =============================================================================

def run_interactive():
    """持续对话模式"""
    print("=" * 60)
    print("试题优化 Agent - 智能对话模式")
    print("=" * 60)
    print("""
这是一个智能 Agent，可以理解你的意图并调用对应工具。

你说的话会被分析，然后 Agent 会决定：
- 如果要提交试题 → 调用 submit_question
- 如果要优化 → 调用 optimize_question
- 如果要修改 → 调用 modify_question
- 如果只是聊天 → 调用 chat

开始对话吧！
""")

    print("\n初始化 Agent...")
    try:
        optimizer = QuestionOptimizer(provider="deepseek")
        print("✅ Agent 创建成功!\n")
    except ValueError as e:
        print(f"❌ Agent 初始化失败: {e}")
        return

    current_question = None
    conversation_history = []

    print("=" * 40)
    print("开始对话（输入 quit 退出）")
    print("=" * 40)

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\n再见！")
                break

            # 分析意图并路由
            router = ToolRouter(optimizer, current_question)
            decision = router.route(user_input)

            tool_name = decision.get("tool", "chat")
            args = decision.get("args", {})

            # 执行工具
            result, new_question = execute_tool(tool_name, args, optimizer, current_question)

            if new_question is not None and new_question != current_question:
                current_question = new_question

            # 输出结果
            if result["success"]:
                if tool_name in ["submit_question", "optimize_question"]:
                    # 显示评估结果
                    data = result.get("data", {})
                    if "result" in data:
                        r = data["result"]
                        print(f"\n📊 {result['message']}")
                        for h in r['history']:
                            scores = ", ".join([f"{k}={v:.3f}" for k, v in h['scores'].items()])
                            print(f"   迭代{h['iteration']}: avg={h['avg']:.3f} [{scores}]")

                        # 输出完整试题
                        best_q = r['best_question']
                        print(f"\n📝 最佳试题:")
                        print(f"   {best_q.question}")
                        print(f"   {best_q.options}")
                        print(f"   答案: {best_q.answer}")
                        if best_q.features.get("knowledge_point"):
                            print(f"   知识点: {best_q.features['knowledge_point']}")
                elif tool_name == "modify_question":
                    print(f"\n✅ {result['message']}")
                    if result["data"] and "new_question" in result["data"]:
                        new_q = result["data"]["new_question"]
                        print(f"   {new_q.question}")
                        print(f"   {new_q.options}")
                        print(f"   答案: {new_q.answer}")
                elif tool_name == "chat":
                    print(f"\n🤖 Agent: {result['message']}")
                else:
                    print(f"\n✅ {result['message']}")
            else:
                print(f"\n❌ {result['message']}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        print("""
Question Optimization Agent
===========================

交互模式（推荐）:
    python question_optimizer_agent.py --interactive

输入格式：用 ; 分隔
示例：
    在我国西北干旱地区...被称为（    ）。;A. 风蚀柱 B. 风蚀蘑菇 C. 风蚀城堡 D. 风蚀洼地;B;风蚀地貌包含风蚀蘑菇。;风成地貌;简单;风蚀蘑菇

字段顺序：题干;选项;答案;知识上下文;章节;难度;知识点

Python 代码使用:
    from question_optimizer_agent import QuestionOptimizer, QuestionData
""")