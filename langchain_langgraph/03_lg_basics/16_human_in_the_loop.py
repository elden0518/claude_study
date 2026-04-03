"""
主题：Human-in-the-Loop —— 让 AI 暂停等待人工审核

学习目标：
  1. 理解 Human-in-the-Loop 的应用场景（高风险操作、内容审核）
  2. 掌握 interrupt() 函数暂停图执行
  3. 理解 MemorySaver（interrupt 必须有 checkpointer 才能工作）
  4. 掌握 Command(resume=...) 恢复图执行
  5. 学会 graph.get_state() 查看暂停时的状态

核心概念：
  interrupt() = 暂停图，等待外部输入后再继续
  必须条件：graph.compile(checkpointer=MemorySaver())
  thread_id  = 标识一次完整的对话/执行流程

  流程：
  第一次 invoke → 遇到 interrupt → 抛出异常，图暂停
  检查状态（可选）
  第二次 invoke(Command(resume=用户输入)) → 从 interrupt 处继续执行

前置知识：已完成 15_cycles_loops.py
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
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

MODEL = "ppio/pa/claude-sonnet-4-6"
llm = ChatAnthropic(model=MODEL, max_tokens=256)


# =============================================================================
# Part 1：基础 interrupt 演示
# =============================================================================

class ContentState(TypedDict):
    topic: str
    draft: str
    feedback: str
    published: bool


def draft_node(state: ContentState) -> dict:
    """生成草稿"""
    print(f"  [draft_node] 生成关于「{state['topic']}」的草稿...")
    response = llm.invoke(f"写一段关于「{state['topic']}」的简短介绍（80字内）")
    return {"draft": response.content}


def review_node(state: ContentState) -> dict:
    """审核节点：暂停图执行，等待人工输入"""
    print(f"  [review_node] 草稿已生成，等待人工审核...")

    # interrupt() 暂停图，value 参数作为提示信息传递给外部
    # 恢复时，interrupt() 的返回值 = Command(resume=...) 中传入的值
    feedback = interrupt(
        f"请审核以下草稿，输入反馈或输入'通过'：\n\n{state['draft']}"
    )

    print(f"  [review_node] 收到审核反馈: {feedback}")
    return {"feedback": feedback}


def publish_node(state: ContentState) -> dict:
    """发布节点：根据反馈决定直接发布或修改后发布"""
    if state["feedback"] == "通过":
        print(f"  [publish_node] 审核通过，直接发布")
        final_content = state["draft"]
    else:
        print(f"  [publish_node] 根据反馈修改后发布")
        response = llm.invoke(
            f"根据反馈修改文章：\n原文：{state['draft']}\n反馈：{state['feedback']}"
        )
        final_content = response.content

    print(f"  [publish_node] 已发布: {final_content[:60]}...")
    return {"draft": final_content, "published": True}


def demo_basic_interrupt():
    """演示基础 interrupt 流程"""
    graph = StateGraph(ContentState)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_node("publish", publish_node)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", "publish")
    graph.add_edge("publish", END)

    # interrupt 必须配合 checkpointer 使用
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "basic-interrupt-demo"}}

    print("  步骤1：第一次 invoke（将在 review_node 的 interrupt 处暂停）")
    try:
        result = app.invoke(
            {"topic": "人工智能", "draft": "", "feedback": "", "published": False},
            config
        )
    except Exception as e:
        # LangGraph 的 interrupt 在某些版本会通过异常机制暂停
        pass

    print("\n  步骤2：图已暂停，检查当前状态")
    state_snapshot = app.get_state(config)
    print(f"  当前草稿: {state_snapshot.values.get('draft', '')[:60]}...")
    print(f"  下一步节点: {state_snapshot.next}")

    print("\n  步骤3：使用 Command(resume=...) 恢复执行（模拟人工输入'通过'）")
    final_result = app.invoke(Command(resume="通过"), config)
    print(f"\n  [结果] published: {final_result.get('published', False)}")


# =============================================================================
# Part 2：暂停时查看状态详情
# =============================================================================

class ArticleState(TypedDict):
    title: str
    content: str
    reviewer_comment: str
    status: str   # "drafting" / "pending_review" / "approved" / "rejected"


def write_article_node(state: ArticleState) -> dict:
    """写文章节点"""
    print(f"  [write_article_node] 撰写文章: {state['title']}")
    response = llm.invoke(f"写一篇关于「{state['title']}」的文章（150字内）")
    return {"content": response.content, "status": "pending_review"}


def human_review_node(state: ArticleState) -> dict:
    """人工审核节点：暂停等待审核意见"""
    print(f"  [human_review_node] 文章待审核，调用 interrupt...")
    comment = interrupt({
        "message": "请审核以下文章",
        "title": state["title"],
        "content": state["content"],
        "instruction": "输入 'approve' 批准，或输入具体修改意见",
    })
    return {"reviewer_comment": str(comment)}


def finalize_node(state: ArticleState) -> dict:
    """根据审核意见最终处理"""
    if state["reviewer_comment"].lower() in ["approve", "通过", "ok"]:
        print(f"  [finalize_node] 文章已批准发布")
        return {"status": "approved"}
    else:
        print(f"  [finalize_node] 文章需要修改，状态: rejected")
        return {"status": "rejected"}


def demo_state_inspection():
    """演示暂停时查看完整状态"""
    graph = StateGraph(ArticleState)
    graph.add_node("write", write_article_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "write")
    graph.add_edge("write", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "state-inspection-demo"}}

    print("  触发 interrupt，图暂停...")
    try:
        app.invoke(
            {"title": "深度学习入门", "content": "", "reviewer_comment": "", "status": "drafting"},
            config
        )
    except Exception:
        pass

    # 查看暂停时的详细状态
    snapshot = app.get_state(config)
    print(f"\n  [暂停时状态快照]")
    print(f"  标题    : {snapshot.values.get('title', '')}")
    print(f"  内容预览: {snapshot.values.get('content', '')[:60]}...")
    print(f"  当前状态: {snapshot.values.get('status', '')}")
    print(f"  下一节点: {snapshot.next}")
    print(f"  配置信息: thread_id = {snapshot.config.get('configurable', {}).get('thread_id', '')}")

    print("\n  以 'approve' 恢复执行...")
    final = app.invoke(Command(resume="approve"), config)
    print(f"  最终状态: {final.get('status', '')}")


# =============================================================================
# Part 3：完整邮件审核工作流
# =============================================================================

class EmailState(TypedDict):
    recipient: str
    subject: str
    body: str
    human_decision: str   # "send" / "revise" / "cancel"
    sent: bool


def compose_email_node(state: EmailState) -> dict:
    """撰写邮件"""
    print(f"  [compose_email_node] 撰写发给 {state['recipient']} 的邮件")
    response = llm.invoke(
        f"写一封关于「{state['subject']}」的简短邮件（50字内），收件人：{state['recipient']}"
    )
    return {"body": response.content}


def approve_email_node(state: EmailState) -> dict:
    """邮件审核：暂停等待人工决策"""
    print(f"  [approve_email_node] 邮件草稿待审核，暂停执行...")
    decision = interrupt(
        f"邮件草稿：\n收件人：{state['recipient']}\n主题：{state['subject']}\n正文：{state['body']}\n\n"
        f"请输入决策：send（发送）/ revise（修改）/ cancel（取消）"
    )
    return {"human_decision": str(decision)}


def send_or_cancel_node(state: EmailState) -> dict:
    """根据人工决策执行发送或取消"""
    decision = state["human_decision"].lower().strip()
    if decision == "send":
        print(f"  [send_or_cancel_node] 邮件已发送给 {state['recipient']}")
        return {"sent": True}
    elif decision == "cancel":
        print(f"  [send_or_cancel_node] 邮件已取消")
        return {"sent": False}
    else:
        # revise 或其他反馈：修改后发送
        print(f"  [send_or_cancel_node] 修改邮件后发送...")
        response = llm.invoke(f"根据反馈修改邮件：\n原文：{state['body']}\n反馈：{decision}")
        print(f"  修改后正文: {response.content[:60]}...")
        return {"body": response.content, "sent": True}


def demo_email_workflow():
    """演示完整的邮件审核工作流"""
    graph = StateGraph(EmailState)
    graph.add_node("compose", compose_email_node)
    graph.add_node("approve", approve_email_node)
    graph.add_node("execute", send_or_cancel_node)

    graph.add_edge(START, "compose")
    graph.add_edge("compose", "approve")
    graph.add_edge("approve", "execute")
    graph.add_edge("execute", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "email-workflow-demo"}}

    # 第一次执行（会在 approve 节点暂停）
    try:
        app.invoke({
            "recipient": "boss@company.com",
            "subject": "项目进展报告",
            "body": "",
            "human_decision": "",
            "sent": False,
        }, config)
    except Exception:
        pass

    print(f"  图已暂停，等待人工审核...")
    state_now = app.get_state(config)
    print(f"  邮件正文预览: {state_now.values.get('body', '')[:60]}...")

    # 模拟人工点击"发送"
    print(f"\n  模拟人工操作：输入 'send'")
    final = app.invoke(Command(resume="send"), config)
    print(f"  邮件已发送: {final.get('sent', False)}")


def main():
    print("=" * 60)
    print("Part 1：基础 interrupt 演示")
    print("=" * 60)
    demo_basic_interrupt()

    print("\n" + "=" * 60)
    print("Part 2：暂停时查看状态详情")
    print("=" * 60)
    demo_state_inspection()

    print("\n" + "=" * 60)
    print("Part 3：完整邮件审核工作流")
    print("=" * 60)
    demo_email_workflow()


if __name__ == "__main__":
    main()
