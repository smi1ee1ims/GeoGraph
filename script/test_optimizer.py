"""
测试脚本 - 如何使用 Question Optimizer Agent
==============================================

前置条件：
1. 已激活 conda 环境: conda activate lc3
2. Neo4j 数据库运行中（bolt://localhost:7689）
3. 题库文件存在: ./data/题库 _已标注.xlsx

运行方式：
    python test_optimizer.py

注意：如果在基础 Python 环境中运行遇到 pydantic 相关错误，
    请先切换到 conda 环境: conda activate lc3
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# 配置常量
# =============================================================================

THRESHOLDS = {
    "relevance": 0.65,
    "explainability": 0.6,
    "difficulty": 0.60
}
MAX_ITERATIONS = 3


@dataclass
class QuestionData:
    """试题数据结构"""
    question: str
    options: str
    answer: str
    knowledge_context: str
    features: dict
    target_difficulty: str = "中等"


@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    metric: str
    score: float
    passed: bool
    details: str
    improvement_hints: list


# =============================================================================
# 测试 1: 直接使用 EvaluatorSubagent 评估单个试题
# =============================================================================

def test_single_evaluator():
    """测试单个评估器"""
    print("\n" + "="*60)
    print("测试 1: 单个评估器评估")
    print("="*60)

    # 模拟评估函数
    def mock_relevance_func(question, context):
        """模拟的知识相关性评估"""
        return 0.75

    # 模拟评估器类（不依赖外部库）
    class MockEvaluator:
        def __init__(self, metric_name, evaluate_func, threshold):
            self.metric = metric_name
            self.evaluate_func = evaluate_func
            self.threshold = threshold

        def evaluate(self, question_data):
            score = self.evaluate_func(
                question_data["question"],
                question_data["knowledge_context"]
            )
            passed = score >= self.threshold
            hints = []
            if not passed:
                hints.append(f"增加与{self.metric}相关的描述")
            return EvaluationResult(
                metric=self.metric,
                score=score,
                passed=passed,
                details=f"{self.metric}: {score:.3f} >= {self.threshold}",
                improvement_hints=hints
            )

    evaluator = MockEvaluator(
        metric_name="relevance",
        evaluate_func=mock_relevance_func,
        threshold=THRESHOLDS["relevance"]
    )

    question_data = {
        "question": "虎跳峡位于中国西南地区，是世界上最深的峡谷之一。该地貌的形成与下列哪一因素关系最为密切？",
        "knowledge_context": "虎跳峡位于中国西南地区。流水侵蚀地貌形成于虎跳峡。虎跳峡是典型流水侵蚀地貌。",
    }

    result = evaluator.evaluate(question_data)

    print(f"\n评估结果:")
    print(f"  指标: {result.metric}")
    print(f"  分数: {result.score:.3f}")
    print(f"  通过: {'是' if result.passed else '否'}")
    print(f"  详情: {result.details}")
    print(f"  改进建议: {result.improvement_hints}")

    return result


# =============================================================================
# 测试 2: 准备输入数据 - QuestionData 结构
# =============================================================================

def test_question_data():
    """测试 QuestionData 结构"""
    print("\n" + "="*60)
    print("测试 2: QuestionData 数据结构")
    print("="*60)

    # 创建一个试题数据对象
    question = QuestionData(
        question="虎跳峡位于中国西南地区，是世界上最深的峡谷之一。该地貌的形成与下列哪一因素关系最为密切？",
        options="A. 冰川的刨蚀作用\nB. 地壳的持续抬升\nC. 流水的溯源侵蚀\nD. 风力的长期吹蚀",
        answer="C",
        knowledge_context="虎跳峡位于中国西南地区。流水侵蚀地貌形成于虎跳峡。虎跳峡是典型流水侵蚀地貌。",
        features={
            "section": "流水地貌",
            "bank_path": "./data/题库 _已标注.xlsx",
            "knowledge_point": "流水侵蚀地貌"
        },
        target_difficulty="中等"
    )

    print(f"\n试题数据:")
    print(f"  题干: {question.question[:50]}...")
    print(f"  选项: {question.options[:50]}...")
    print(f"  答案: {question.answer}")
    print(f"  知识上下文: {question.knowledge_context[:50]}...")
    print(f"  目标难度: {question.target_difficulty}")
    print(f"  章节: {question.features.get('section')}")

    return question


# =============================================================================
# 测试 3: 优化决策逻辑
# =============================================================================

def test_optimization_decision():
    """测试优化决策逻辑"""
    print("\n" + "="*60)
    print("测试 3: 优化决策逻辑")
    print("="*60)

    # 模拟评估结果
    test_cases = [
        {"relevance": 0.72, "explainability": 0.85, "difficulty": 0.65},  # 全部通过
        {"relevance": 0.55, "explainability": 0.85, "difficulty": 0.65},  # 1个失败
        {"relevance": 0.55, "explainability": 0.50, "difficulty": 0.65},  # 2个失败
        {"relevance": 0.55, "explainability": 0.50, "difficulty": 0.45},  # 3个失败
    ]

    for i, scores in enumerate(test_cases):
        print(f"\n案例 {i+1}: {scores}")

        # 判断每个指标是否通过
        results = {
            "relevance": scores["relevance"] >= THRESHOLDS["relevance"],
            "explainability": scores["explainability"] >= THRESHOLDS["explainability"],
            "difficulty": scores["difficulty"] >= THRESHOLDS["difficulty"]
        }

        failing = [k for k, v in results.items() if not v]
        avg_score = sum(scores.values()) / len(scores)

        print(f"  评估结果: {results}")
        print(f"  失败指标: {failing}")
        print(f"  平均分: {avg_score:.3f}")

        if len(failing) == 0:
            print("  策略: 全部通过，无需优化")
        elif len(failing) == 1:
            print(f"  策略: 单目标优化 ({failing[0]})")
        elif len(failing) >= 2:
            print(f"  策略: 多目标优化 (同时优化: {', '.join(failing)})")


# =============================================================================
# 测试 4: 并行评估模拟
# =============================================================================

def test_parallel_evaluation():
    """测试并行评估逻辑"""
    print("\n" + "="*60)
    print("测试 4: 并行评估逻辑演示")
    print("="*60)

    import time
    from concurrent.futures import ThreadPoolExecutor

    def mock_evaluate(metric_name, delay=0.1):
        """模拟评估（带延迟）"""
        time.sleep(delay)
        scores = {"relevance": 0.72, "explainability": 0.85, "difficulty": 0.65}
        return metric_name, scores.get(metric_name, 0.5)

    print("\n执行 3 个评估器并行评估...")

    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(mock_evaluate, "relevance", 0.2),
            executor.submit(mock_evaluate, "explainability", 0.15),
            executor.submit(mock_evaluate, "difficulty", 0.1)
        ]

        results = {}
        for future in futures:
            metric, score = future.result()
            results[metric] = score
            print(f"  {metric}: {score:.3f}")

    elapsed = time.time() - start
    avg_score = sum(results.values()) / len(results)

    print(f"\n总耗时: {elapsed:.3f}秒（并行）")
    print(f"平均分: {avg_score:.3f}")

    return results


# =============================================================================
# 测试 5: 迭代优化流程
# =============================================================================

def test_iteration_flow():
    """测试迭代优化流程"""
    print("\n" + "="*60)
    print("测试 5: 迭代优化流程")
    print("="*60)

    print("""
优化流程示意:

Q0 (初始试题)
    │
    ▼
┌─────────────────────────────┐
│ 并行评估 (R, E, D)          │
└─────────────────────────────┘
    │
    ▼
失败数 = N
    │
    ├─ N=0 → 全部通过 → 返回 Q
    ├─ N=1 → 单目标优化 → Q1
    └─ N>=2 → 多目标优化 → Q1
    │
    ▼
i < max_iterations?
    │
    ├─ 是 → 返回评估循环
    └─ 否 → 返回 best_question
""")


# =============================================================================
# 测试 6: 实际使用示例
# =============================================================================

def show_real_usage_example():
    """展示实际使用方式"""
    print("\n" + "="*60)
    print("测试 6: 实际使用示例代码")
    print("="*60)

    example_code = '''
# =====================================
# 实际使用时，从 eval.ipynb 导入评估函数
# =====================================

# 方法: 在 Jupyter Notebook 中使用 %paste 或手动复制评估函数

# 需要导入的函数（来自 eval.ipynb）:
# - compute_knowledge_relevance(full_question, knowledge_context) -> float
# - eval_difficulty_accuracy(generated_txt, bank_xlsx, section) -> float
# - eval_answer(input_txt_path, output_txt_path, n) 用于可解释性评估

# =====================================
# 完整使用示例
# =====================================

from question_optimizer_agent import create_question_optimizer, QuestionData

# 1. 创建优化器（需要 API key 和评估函数）
optimizer = create_question_optimizer(
    anthropic_api_key="your-anthropic-api-key",
    evaluator_funcs={
        "relevance": compute_knowledge_relevance,      # 来自 eval.ipynb
        "explainability": compute_explainability,       # 来自 eval.ipynb
        "difficulty": eval_difficulty_accuracy         # 来自 eval.ipynb
    }
)

# 2. 准备试题数据（来自 graphRAG 生成结果）
question = QuestionData(
    question="生成的试题题干...",
    options="A. 选项1\\nB. 选项2\\nC. 选项3\\nD. 选项4",
    answer="A",
    knowledge_context="知识上下文...",
    features={
        "section": "流水地貌",
        "bank_path": "./data/题库 _已标注.xlsx"
    },
    target_difficulty="中等"
)

# 3. 运行优化
result = optimizer.run_optimization(question)

# 4. 获取结果
print(f"迭代次数: {result.iterations}")
print(f"全部通过: {result.all_passed}")
print(f"最佳平均分: {result.best_avg_score:.3f}")
print(f"最佳试题: {result.best_question.question}")

# 5. 查看评估历史
for entry in result.evaluation_history:
    print(f"迭代 {entry['iteration']}: avg={entry['avg_score']:.3f}")
'''

    print(example_code)


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*60)
    print("Question Optimizer Agent 测试")
    print("="*60)

    # 运行所有测试
    test_single_evaluator()
    test_question_data()
    test_optimization_decision()
    test_parallel_evaluation()
    test_iteration_flow()
    show_real_usage_example()

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print("""
下一步：
1. 确保 conda activate lc3 环境已激活
2. 确保 Neo4j 数据库运行中
3. 确保题库文件存在于 ./data/题库 _已标注.xlsx
4. 提供有效的 API key（ANTHROPIC_API_KEY 或 DEEPSEEK_API_KEY）
5. 根据 show_real_usage_example() 中的代码进行实际使用
""")


if __name__ == "__main__":
    main()