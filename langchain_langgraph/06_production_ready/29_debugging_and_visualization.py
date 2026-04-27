"""
主题：LangGraph 调试与可视化 —— 让复杂工作流一目了然

学习目标：
  1. 理解为什么需要可视化工具（复杂图难以理解）
  2. 掌握 get_graph().draw_mermaid() 导出 Mermaid 图
  3. 掌握 LangGraph Studio（官方可视化工具）
  4. 学会使用 debug 模式追踪执行路径
  5. 掌握状态快照检查和历史回放

核心概念：
  可视化 = 将图的节点和边转换为图形表示
  Debug = 实时查看每个节点的输入输出
  State Snapshot = 某个时间点的完整状态快照
  
  工具选择：
  - Mermaid：快速生成静态图（适合文档）
  - LangGraph Studio：交互式调试（适合开发）
  - Stream Events：编程式追踪（适合监控）

前置知识：已完成 03_lg_basics/13_hello_langgraph.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=128)


# =============================================================================
# Part 1：导出 Mermaid 图（静态可视化）
# =============================================================================

def demo_export_mermaid():
    """
    LangGraph 内置支持导出为 Mermaid 格式，可渲染为流程图。
    
    优点：
    - 无需额外工具，代码即可生成
    - 可嵌入 Markdown 文档
    - GitHub、Notion 等原生支持 Mermaid
    
    使用场景：
    - 技术文档中的架构图
    - Code Review 时展示逻辑
    - 团队分享工作流设计
    """
    print("=" * 60)
    print("Part 1: Export to Mermaid Diagram")
    print("=" * 60)
    
    # 定义一个简单的多节点图
    class SimpleState(TypedDict):
        messages: Annotated[list, add_messages]
        category: str
    
    def classify_node(state: SimpleState) -> dict:
        """分类节点"""
        return {"category": "technical"}
    
    def respond_node(state: SimpleState) -> dict:
        """回复节点"""
        response = llm.invoke(f"回答用户问题：{state['messages'][-1].content}")
        return {"messages": [AIMessage(content=response.content)]}
    
    # 构建图
    graph = StateGraph(SimpleState)
    graph.add_node("classify", classify_node)
    graph.add_node("respond", respond_node)
    
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    
    app = graph.compile()
    
    # 导出为 Mermaid
    print("\n【Mermaid 图代码】")
    print("-" * 60)
    mermaid_code = app.get_graph().draw_mermaid()
    print(mermaid_code)
    print("-" * 60)
    
    print("\n💡 使用方法：")
    print("  1. 复制上面的代码")
    print("  2. 粘贴到 https://mermaid.live 查看效果")
    print("  3. 或在 Markdown 中使用：```mermaid ... ```")
    
    # 保存为文件
    output_path = "./graph_visualization.mmd"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)
    print(f"\n✅ 已保存到：{output_path}")


# =============================================================================
# Part 2：Debug 模式 —— 实时追踪执行
# =============================================================================

def demo_debug_mode():
    """
    LangGraph 支持详细的调试输出，显示每个节点的执行过程。
    
    启用方式：
    1. 设置环境变量：LANGGRAPH_DEBUG=true
    2. 或在 invoke 时传入 config：config={"debug": True}
    
    输出信息：
    - 节点开始/结束时间
    - 输入状态的快照
    - 输出状态的变更
    - 条件边的路由决策
    """
    print("\n" + "=" * 60)
    print("Part 2: Debug Mode —— 实时追踪")
    print("=" * 60)
    
    class DebugState(TypedDict):
        step: int
        data: str
    
    def step1(state: DebugState) -> dict:
        print(f"  [step1] 输入：step={state['step']}, data={state['data']}")
        return {"step": state["step"] + 1, "data": state["data"] + " → processed by step1"}
    
    def step2(state: DebugState) -> dict:
        print(f"  [step2] 输入：step={state['step']}, data={state['data']}")
        return {"step": state["step"] + 1, "data": state["data"] + " → processed by step2"}
    
    # 构建图
    graph = StateGraph(DebugState)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)
    
    app = graph.compile()
    
    print("\n【执行追踪】")
    result = app.invoke({"step": 0, "data": "initial"})
    
    print(f"\n【最终状态】")
    print(f"  step: {result['step']}")
    print(f"  data: {result['data']}")
    
    print("\n💡 提示：生产环境可使用 LangSmith 替代手动 debug")


# =============================================================================
# Part 3：状态快照与历史回放
# =============================================================================

def demo_state_snapshots():
    """
    通过 Checkpointer 保存状态快照，支持：
    - 查看任意时刻的状态
    - 从历史状态恢复执行
    - 调试时回溯问题根源
    """
    print("\n" + "=" * 60)
    print("Part 3: State Snapshots & Replay")
    print("=" * 60)
    
    from langgraph.checkpoint.memory import MemorySaver
    
    class WorkflowState(TypedDict):
        counter: int
        log: list
    
    def increment(state: WorkflowState) -> dict:
        new_counter = state["counter"] + 1
        new_log = state["log"] + [f"Step {new_counter}: incremented"]
        print(f"  [increment] counter={new_counter}")
        return {"counter": new_counter, "log": new_log}
    
    # 构建带 checkpoint 的图
    graph = StateGraph(WorkflowState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "snapshot-demo"}}
    
    # 第一次执行
    print("\n【第 1 次执行】")
    result1 = app.invoke({"counter": 0, "log": []}, config)
    print(f"  结果：counter={result1['counter']}")
    
    # 获取状态快照
    print("\n【状态快照】")
    snapshot = app.get_state(config)
    print(f"  当前值：{snapshot.values}")
    print(f"  下一节点：{snapshot.next}")
    print(f"  检查点 ID：{snapshot.checkpoint_id}")
    
    # 查看所有检查点历史
    print("\n【检查点历史】")
    checkpoints = list(app.get_state_history(config))
    for i, cp in enumerate(checkpoints):
        print(f"  [{i}] checkpoint_id={cp.checkpoint_id[:8]}... | values={cp.values}")
    
    # 从历史状态恢复（可选）
    if len(checkpoints) > 1:
        print("\n【从历史状态恢复】")
        old_checkpoint = checkpoints[0]
        print(f"  恢复到：{old_checkpoint.values}")
        
        # 注意：实际使用时需要构造正确的 config
        # restored_config = {"configurable": {"thread_id": "...", "checkpoint_id": "..."}}
        # result = app.invoke(None, restored_config)


# =============================================================================
# Part 4：LangGraph Studio（交互式可视化工具）
# =============================================================================

def demo_langgraph_studio():
    """
    LangGraph Studio 是官方提供的可视化工具，功能强大：
    
    特性：
    ✅ 实时图形化展示工作流
    ✅ 交互式执行和调试
    ✅ 查看每个节点的状态变化
    ✅ 支持 Human-in-the-Loop 操作
    ✅ 历史记录和回放
    
    安装和启动：
    ──────────────────────────────────────────────
    1. 安装：pip install langgraph-cli
    2. 在项目根目录创建 langgraph.json 配置文件
    3. 运行：langgraph dev
    4. 浏览器访问：http://localhost:2024
    
    配置文件示例（langgraph.json）：
    ──────────────────────────────────────────────
    {
      "dependencies": ["."],
      "graphs": {
        "my_agent": "./src/agent.py:graph"
      },
      "env": ".env"
    }
    """
    print("\n" + "=" * 60)
    print("Part 4: LangGraph Studio —— 交互式可视化")
    print("=" * 60)
    
    guide = """
  📦 安装步骤：
  ──────────────────────────────────────────────
  
  1️⃣  安装 CLI 工具
     $ pip install langgraph-cli
  
  2️⃣  创建配置文件（项目根目录）
     文件名：langgraph.json
     
     内容示例：
     {
       "dependencies": ["."],
       "graphs": {
         "research_team": "./05_multi_agent/24_research_team.py:app",
         "simple_agent": "./03_lg_basics/13_hello_langgraph.py:app"
       },
       "env": ".env"
     }
  
  3️⃣  启动开发服务器
     $ langgraph dev
     
     或指定端口：
     $ langgraph dev --port 2024
  
  4️⃣  浏览器访问
     http://localhost:2024
  
  🎯 主要功能：
  ──────────────────────────────────────────────
  
  • Graph View：可视化展示节点和边的连接关系
  • Thread View：查看每次执行的完整轨迹
  • State Inspector：检查任意时刻的状态详情
  • Interactive Run：手动触发执行并观察结果
  • Breakpoints：在特定节点设置断点
  • Time Travel：从历史状态重新开始
  
  💡 最佳实践：
  ──────────────────────────────────────────────
  
  1. 为每个重要的图添加描述性名称
  2. 使用有意义的节点名称（避免 node1, node2）
  3. 在复杂图中添加注释说明
  4. 定期导出 Mermaid 图作为文档备份
  5. 结合 LangSmith 进行生产环境监控
  
  🔗 相关链接：
  ──────────────────────────────────────────────
  
  • 官方文档：https://langchain-ai.github.io/langgraph/concepts/langgraph_studio
  • GitHub：https://github.com/langchain-ai/langgraph
  • 示例项目：https://github.com/langchain-ai/langgraph/tree/main/examples
    """
    print(guide)


# =============================================================================
# Part 5：综合示例 —— 复杂图的可视化
# =============================================================================

def demo_complex_graph_visualization():
    """
    展示一个包含条件分支、循环、并行的复杂图，并导出可视化。
    """
    print("\n" + "=" * 60)
    print("Part 5: Complex Graph Visualization")
    print("=" * 60)
    
    from typing import Literal
    
    class ComplexState(TypedDict):
        messages: Annotated[list, add_messages]
        iteration: int
        status: str
    
    def preprocess(state: ComplexState) -> dict:
        """预处理节点"""
        return {"status": "preprocessed"}
    
    def analyze(state: ComplexState) -> dict:
        """分析节点"""
        return {"status": "analyzed"}
    
    def decide_next(state: ComplexState) -> Literal["refine", "finalize"]:
        """条件路由"""
        if state["iteration"] < 3:
            return "refine"
        return "finalize"
    
    def refine(state: ComplexState) -> dict:
        """优化节点（循环）"""
        return {"iteration": state["iteration"] + 1, "status": "refined"}
    
    def finalize(state: ComplexState) -> dict:
        """最终化节点"""
        response = llm.invoke("总结整个流程")
        return {
            "messages": [AIMessage(content=response.content)],
            "status": "completed"
        }
    
    # 构建复杂图
    graph = StateGraph(ComplexState)
    graph.add_node("preprocess", preprocess)
    graph.add_node("analyze", analyze)
    graph.add_node("refine", refine)
    graph.add_node("finalize", finalize)
    
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "analyze")
    graph.add_conditional_edges("analyze", decide_next)
    graph.add_edge("refine", "analyze")  # 循环
    graph.add_edge("finalize", END)
    
    app = graph.compile()
    
    # 导出 Mermaid
    print("\n【复杂图 Mermaid 代码】")
    print("-" * 60)
    mermaid_code = app.get_graph().draw_mermaid()
    print(mermaid_code)
    print("-" * 60)
    
    # 保存
    output_path = "./complex_graph_visualization.mmd"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)
    print(f"\n✅ 已保存到：{output_path}")
    
    print("\n💡 这个图包含：")
    print("  • 线性流程：preprocess → analyze")
    print("  • 条件分支：decide_next 路由到 refine 或 finalize")
    print("  • 循环结构：refine → analyze（最多 3 次迭代）")


def main():
    print("=" * 60)
    print("LangGraph 调试与可视化详解")
    print("=" * 60)
    
    demo_export_mermaid()
    demo_debug_mode()
    demo_state_snapshots()
    demo_langgraph_studio()
    demo_complex_graph_visualization()
    
    print("\n" + "=" * 60)
    print("学习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
