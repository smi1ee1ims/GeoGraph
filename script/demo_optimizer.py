"""
用户测试 - 试题优化演示
========================
模拟真实用户使用场景：输入一道试题，体验完整优化过程

运行方式：
    python demo_optimizer.py
"""

import time

# =============================================================================
# 场景设置
# =============================================================================

# 用户输入的试题
USER_QUESTION = {
    "question": "虎跳峡位于中国西南地区，是世界上最深的峡谷之一，以水流湍急、峡谷幽深著称。峡谷两岸的岩层陡峭，谷底狭窄，江心有一巨石（虎跳石）矗立。该地貌的形成与下列哪一因素关系最为密切？",
    "options": "A. 冰川的刨蚀作用\nB. 地壳的持续抬升\nC. 流水的溯源侵蚀\nD. 风力的长期吹蚀",
    "answer": "C",
    "knowledge_context": "虎跳峡位于中国西南地区。流水侵蚀地貌形成于虎跳峡。虎跳峡是典型流水侵蚀地貌。",
    "target_difficulty": "中等"
}

# 初始评估分数（模拟）
INITIAL_SCORES = {
    "relevance": 0.55,       # 低于阈值 0.65
    "explainability": 0.85,   # 通过
    "difficulty": 0.68       # 通过
}

# 阈值
THRESHOLDS = {
    "relevance": 0.65,
    "explainability": 0.6,
    "difficulty": 0.60
}


# =============================================================================
# 模拟评估函数
# =============================================================================

def simulate_evaluate(question_text, knowledge_context, metric):
    """模拟评估过程，返回带评语的分数"""
    time.sleep(0.5)  # 模拟计算时间

    if metric == "relevance":
        score = INITIAL_SCORES["relevance"]
        if score < THRESHOLDS["relevance"]:
            return score, "❌ 未通过", "试题与知识上下文的关联度不够高"
        else:
            return score, "✅ 通过", "试题与知识上下文关联良好"

    elif metric == "explainability":
        score = INITIAL_SCORES["explainability"]
        if score < THRESHOLDS["explainability"]:
            return score, "❌ 未通过", "答案一致性不够稳定"
        else:
            return score, "✅ 通过", "答案一致性良好"

    elif metric == "difficulty":
        score = INITIAL_SCORES["difficulty"]
        if score < THRESHOLDS["difficulty"]:
            return score, "❌ 未通过", "难度与目标不匹配"
        else:
            return score, "✅ 通过", "难度符合目标"


def simulate_optimize(question, failing_metrics):
    """模拟优化过程，返回优化后的试题"""
    time.sleep(1.0)  # 模拟生成时间

    optimized = {
        "relevance": {
            "before": "虎跳峡位于中国西南地区...",
            "after": "虎跳峡位于中国西南地区，河流在漫长的地质时期中持续下切，侵蚀两岸岩石，形成深邃的峡谷。作为典型的流水侵蚀地貌，其形成主要与哪种作用有关？",
            "improvement": "✓ 更明确地关联了'流水侵蚀地貌'这个知识点"
        },
        "explainability": {
            "improvement": "✓ 减少了选项间的歧义，答案更明确"
        },
        "difficulty": {
            "improvement": "✓ 调整了条件复杂度，难度更匹配"
        }
    }

    return optimized


# =============================================================================
# 主演示流程
# =============================================================================

def print_separator():
    print("=" * 60)


def print_question(title, q):
    print(f"\n【{title}】")
    print(f"试题：{q['question'][:80]}...")
    print(f"选项：{q['options']}")
    print(f"答案：{q['answer']}")
    print(f"知识上下文：{q['knowledge_context'][:50]}...")


def main():
    print_separator()
    print("        试题优化 Agent - 用户演示")
    print("        模拟真实优化流程，用户体验展示")
    print_separator()

    print("\n📌 场景说明：")
    print("   假设你是一位地理老师，你手头有一道生成的试题，")
    print("   但你不确定它的质量如何，于是你把它交给优化 Agent")
    print("   看看是否需要优化，以及如何优化。")

    # Step 1: 输入试题
    print_separator()
    print("\n【Step 1】 输入试题")
    print("-" * 40)
    print("用户将以下试题提交给优化 Agent：")
    print_question("原始试题", USER_QUESTION)

    input("\n按回车继续...")

    # Step 2: 并行评估
    print_separator()
    print("\n【Step 2】 并行评估（3个指标同时评估）")
    print("-" * 40)
    print("Agent 正在同时评估 3 个指标...")
    print()

    results = {}
    for metric in ["relevance", "explainability", "difficulty"]:
        score, status, comment = simulate_evaluate(
            USER_QUESTION["question"],
            USER_QUESTION["knowledge_context"],
            metric
        )
        results[metric] = {
            "score": score,
            "status": status,
            "comment": comment
        }
        print(f"  📊 {metric:15s}: {score:.3f}  {status}")
        print(f"     → {comment}")

    # Step 3: 分析结果
    print_separator()
    print("\n【Step 3】 分析评估结果")
    print("-" * 40)

    failing_metrics = []
    passing_metrics = []

    for metric, result in results.items():
        threshold = THRESHOLDS[metric]
        if result["score"] < threshold:
            failing_metrics.append(metric)
            print(f"  ❌ {metric:12s} 未通过 (分数 {result['score']:.3f} < 阈值 {threshold})")
        else:
            passing_metrics.append(metric)
            print(f"  ✅ {metric:12s} 通过   (分数 {result['score']:.3f} >= 阈值 {threshold})")

    print(f"\n  📋 汇总：{len(passing_metrics)}/3 通过，{len(failing_metrics)}/3 未通过")

    # Step 4: 决策
    print_separator()
    print("\n【Step 4】 优化决策")
    print("-" * 40)

    if len(failing_metrics) == 0:
        print("  🎉 所有指标都通过了！无需优化。")
        print(f"  最终试题：{USER_QUESTION['question'][:50]}...")
    else:
        print(f"  📝 需要优化的指标：{', '.join(failing_metrics)}")

        if len(failing_metrics) >= 2:
            print("  🔧 策略：多目标优化（同时优化多个指标）")
        else:
            print(f"  🔧 策略：单目标优化（专注于 {failing_metrics[0]}）")

        input("\n按回车继续优化...")

        # Step 5: 优化
        print_separator()
        print("\n【Step 5】 执行优化")
        print("-" * 40)
        print(f"  正在根据评估反馈优化试题...")
        print()

        optimized = simulate_optimize(USER_QUESTION, failing_metrics)

        for metric in failing_metrics:
            print(f"  📝 {metric} 优化：")
            if metric == "relevance" and "before" in optimized["relevance"]:
                print(f"     原文：{optimized['relevance']['before'][:60]}...")
                print(f"     优化：{optimized['relevance']['after'][:60]}...")
            print(f"     {optimized[metric]['improvement']}")
            print()

        # Step 6: 重新评估
        print_separator()
        print("\n【Step 6】 重新评估（优化后）")
        print("-" * 40)
        print("  Agent 正在重新评估优化后的试题...")

        # 模拟优化后的分数
        NEW_SCORES = {
            "relevance": 0.72,      # 提高了！
            "explainability": 0.88,  # 保持
            "difficulty": 0.70       # 保持
        }

        print()
        for metric in ["relevance", "explainability", "difficulty"]:
            score = NEW_SCORES[metric]
            threshold = THRESHOLDS[metric]
            status = "✅ 通过" if score >= threshold else "❌ 未通过"
            print(f"  📊 {metric:15s}: {score:.3f}  {status}")

        # Step 7: 最终结果
        print_separator()
        print("\n【Step 7】 最终结果")
        print("-" * 40)

        all_pass = all(NEW_SCORES[m] >= THRESHOLDS[m] for m in NEW_SCORES)

        if all_pass:
            print("  🎉 所有指标都通过了！")
            print()
            print("  最终优化试题：")
            print("  " + "-" * 40)
            print("  虎跳峡位于中国西南地区，河流在漫长的地质时期中持续下切，")
            print("  侵蚀两岸岩石，形成深邃的峡谷。作为典型的流水侵蚀地貌，")
            print("  其形成主要与哪种作用有关？")
            print()
            print("  A. 冰川的刨蚀作用")
            print("  B. 地壳的持续抬升")
            print("  C. 流水的下切侵蚀")
            print("  D. 风力的长期吹蚀")
            print()
            print("  答案：C")
        else:
            remaining = [m for m in NEW_SCORES if NEW_SCORES[m] < THRESHOLDS[m]]
            print(f"  ⚠️ 仍有 {len(remaining)} 个指标未通过：{', '.join(remaining)}")
            print("  （演示结束，实际会继续迭代优化）")

    # 迭代历史
    print_separator()
    print("\n【评估历史记录】")
    print("-" * 40)
    print("  迭代 1: relevance=0.55, explainability=0.85, difficulty=0.68 → 1个失败")
    print("  迭代 2: relevance=0.72, explainability=0.88, difficulty=0.70 → 全部通过")
    print()
    print("  📈 最佳平均分：0.767")
    print("  ✅ 优化成功！" if all_pass else "  ⏳ 需要更多迭代")

    print_separator()
    print("\n演示结束！")
    print("=" * 60)
    print("""
下一步（真实使用）：
1. 准备好你的 API key（DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY）
2. 从 eval.ipynb 导入评估函数
3. 运行 question_optimizer_agent.py 中的真实优化流程
""")


if __name__ == "__main__":
    main()