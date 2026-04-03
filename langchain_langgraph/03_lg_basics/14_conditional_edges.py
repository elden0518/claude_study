"""
主题：Conditional Edges —— 根据状态动态决定下一步

学习目标：
  1. 理解条件边与普通边的区别（动态 vs 静态路由）
  2. 掌握 add_conditional_edges 的用法
  3. 学会编写路由函数（接收 State，返回节点名字符串）
  4. 实现"分诊"模式：根据输入类型路由到不同处理节点
  5. 掌握多分支路由（3个以上出口）

核心概念：
  普通边：add_edge("A", "B")  → A 始终跳到 B
  条件边：add_conditional_edges("A", route_fn)
         → route_fn 接收 State，返回下一个节点名

  路由函数签名：def route(state: State) -> str
  返回值 = 要跳转的节点名，或 END

前置知识：已完成 13_hello_langgraph.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：简单二分路由（技术 vs 通用）
# =============================================================================

class RoutingState(TypedDict):
    query: str
    category: str   # "technical" 或 "general"
    response: str


def classify_node(state: RoutingState) -> dict:
    """根据关键词将问题分类为技术类或通用类"""
    query_lower = state["query"].lower()
    if any(kw in query_lower for kw in ["python", "代码", "函数", "类", "算法"]):
        return {"category": "technical"}
    return {"category": "general"}


def tech_handler(state: RoutingState) -> dict:
    """处理技术类问题"""
    print(f"  [tech_handler] 处理技术问题: {state['query']}")
    response = llm.invoke(
        f"你是一位技术专家，请简洁地回答这个技术问题：{state['query']}"
    )
    return {"response": response.content}


def general_handler(state: RoutingState) -> dict:
    """处理通用类问题"""
    print(f"  [general_handler] 处理通用问题: {state['query']}")
    response = llm.invoke(
        f"请友好地回答这个问题：{state['query']}"
    )
    return {"response": response.content}


def route_by_category(state: RoutingState) -> str:
    """路由函数：根据 category 决定下一个节点"""
    print(f"  [route_by_category] 分类结果: {state['category']}")
    return "tech_handler" if state["category"] == "technical" else "general_handler"


def demo_binary_routing():
    """演示二分路由：分类节点 → 路由 → 技术处理 / 通用处理"""
    graph = StateGraph(RoutingState)

    # 添加节点
    graph.add_node("classify", classify_node)
    graph.add_node("tech_handler", tech_handler)
    graph.add_node("general_handler", general_handler)

    # 添加普通边
    graph.add_edge(START, "classify")

    # 添加条件边：从 classify 出发，由 route_by_category 决定去哪
    graph.add_conditional_edges("classify", route_by_category)

    # 两个处理节点都通向 END
    graph.add_edge("tech_handler", END)
    graph.add_edge("general_handler", END)

    app = graph.compile()

    # 测试技术问题
    print("  --- 技术问题 ---")
    result1 = app.invoke({
        "query": "Python 中的装饰器是什么？",
        "category": "",
        "response": "",
    })
    print(f"  分类: {result1['category']}")
    print(f"  回答: {result1['response'][:80]}...")

    # 测试通用问题
    print("\n  --- 通用问题 ---")
    result2 = app.invoke({
        "query": "今天天气怎么样？",
        "category": "",
        "response": "",
    })
    print(f"  分类: {result2['category']}")
    print(f"  回答: {result2['response'][:80]}...")


# =============================================================================
# Part 2：三路路由（编程语言专家分流）
# =============================================================================

class ExpertRoutingState(TypedDict):
    query: str
    expert_type: str   # "python" / "java" / "general"
    response: str


def detect_language_node(state: ExpertRoutingState) -> dict:
    """检测问题涉及的编程语言"""
    query_lower = state["query"].lower()
    if "python" in query_lower or "pip" in query_lower or "django" in query_lower:
        return {"expert_type": "python"}
    elif "java" in query_lower or "maven" in query_lower or "spring" in query_lower:
        return {"expert_type": "java"}
    else:
        return {"expert_type": "general"}


def python_expert(state: ExpertRoutingState) -> dict:
    """Python 专家节点"""
    print(f"  [python_expert] 回答 Python 问题")
    response = llm.invoke(
        f"你是 Python 专家，请简洁回答：{state['query']}"
    )
    return {"response": response.content}


def java_expert(state: ExpertRoutingState) -> dict:
    """Java 专家节点"""
    print(f"  [java_expert] 回答 Java 问题")
    response = llm.invoke(
        f"你是 Java 专家，请简洁回答：{state['query']}"
    )
    return {"response": response.content}


def general_expert(state: ExpertRoutingState) -> dict:
    """通用专家节点"""
    print(f"  [general_expert] 回答通用问题")
    response = llm.invoke(
        f"你是全栈专家，请简洁回答：{state['query']}"
    )
    return {"response": response.content}


def route_to_expert(state: ExpertRoutingState) -> str:
    """三路路由函数"""
    mapping = {
        "python": "python_expert",
        "java": "java_expert",
        "general": "general_expert",
    }
    dest = mapping.get(state["expert_type"], "general_expert")
    print(f"  [route_to_expert] 路由到: {dest}")
    return dest


def demo_three_way_routing():
    """演示三路路由：根据编程语言路由到对应专家"""
    graph = StateGraph(ExpertRoutingState)

    graph.add_node("detect", detect_language_node)
    graph.add_node("python_expert", python_expert)
    graph.add_node("java_expert", java_expert)
    graph.add_node("general_expert", general_expert)

    graph.add_edge(START, "detect")

    # 使用 dict 映射显式指定路由出口（可选但更清晰）
    graph.add_conditional_edges(
        "detect",
        route_to_expert,
        {
            "python_expert": "python_expert",
            "java_expert": "java_expert",
            "general_expert": "general_expert",
        }
    )

    graph.add_edge("python_expert", END)
    graph.add_edge("java_expert", END)
    graph.add_edge("general_expert", END)

    app = graph.compile()

    queries = [
        "Python 的 list comprehension 怎么用？",
        "Java 的 Spring Boot 如何配置数据源？",
        "什么是 REST API？",
    ]

    for q in queries:
        print(f"\n  查询: {q}")
        result = app.invoke({"query": q, "expert_type": "", "response": ""})
        print(f"  专家类型: {result['expert_type']}")
        print(f"  回答: {result['response'][:60]}...")


# =============================================================================
# Part 3：路由到 END（提前终止图执行）
# =============================================================================

class ValidatedState(TypedDict):
    query: str
    is_valid: bool
    response: str


def validate_node(state: ValidatedState) -> dict:
    """验证节点：检查输入是否有效"""
    is_valid = bool(state["query"] and state["query"].strip())
    print(f"  [validate_node] 输入{'有效' if is_valid else '无效'}")
    return {"is_valid": is_valid}


def process_valid_query(state: ValidatedState) -> dict:
    """处理有效输入"""
    print(f"  [process_valid_query] 处理: {state['query']}")
    response = llm.invoke(f"简洁回答：{state['query']}")
    return {"response": response.content}


def route_after_validate(state: ValidatedState) -> str:
    """验证后的路由：无效输入直接返回 END，跳过后续处理"""
    if not state["is_valid"]:
        print(f"  [route_after_validate] 输入无效，直接终止")
        return END  # 直接返回 END，图立即停止
    return "process"


def demo_route_to_end():
    """演示路由函数返回 END，实现提前终止"""
    graph = StateGraph(ValidatedState)

    graph.add_node("validate", validate_node)
    graph.add_node("process", process_valid_query)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", route_after_validate)
    graph.add_edge("process", END)

    app = graph.compile()

    # 测试有效输入
    print("  --- 有效输入 ---")
    result1 = app.invoke({"query": "什么是机器学习？", "is_valid": False, "response": ""})
    print(f"  is_valid: {result1['is_valid']}")
    print(f"  response: {result1['response'][:60]}...")

    # 测试空输入（提前终止）
    print("\n  --- 空输入（提前终止）---")
    result2 = app.invoke({"query": "", "is_valid": False, "response": ""})
    print(f"  is_valid: {result2['is_valid']}")
    print(f"  response: '{result2['response']}' (空，因为直接终止了)")


def main():
    print("=" * 60)
    print("Part 1：简单二分路由（技术 vs 通用）")
    print("=" * 60)
    demo_binary_routing()

    print("\n" + "=" * 60)
    print("Part 2：三路路由（编程语言专家分流）")
    print("=" * 60)
    demo_three_way_routing()

    print("\n" + "=" * 60)
    print("Part 3：路由到 END（提前终止图执行）")
    print("=" * 60)
    demo_route_to_end()


if __name__ == "__main__":
    main()
