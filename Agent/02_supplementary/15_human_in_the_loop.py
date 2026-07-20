"""
==============================================================================
第十五课（补充）：Human-in-the-Loop（人机协作）
==============================================================================

【为什么需要单独一课？】
现有课程没有涉及人工干预机制。
生产级 Agent 必须在关键节点引入人工审核，确保安全可控。

【学习目标】
- 理解 Human-in-the-Loop 的必要性
- 掌握工具调用前的人工审批
- 学会实现中断与恢复机制
- 掌握人工反馈循环
- 理解不同级别的干预策略

【核心概念】
- 人工审批（Approval）
- 中断与恢复（Interrupt & Resume）
- 反馈循环（Feedback Loop）
- 干预级别（Intervention Levels）

==============================================================================
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any

from pydantic import BaseModel, Field


# ============================================================================
# 第一部分：为什么需要 Human-in-the-Loop？
# ============================================================================

def why_human_in_the_loop():
    """
    Human-in-the-Loop 的必要性

    Agent 自主性带来的风险：
    1. 可能执行危险操作（删除数据、发送邮件）
    2. 可能产生错误结果（幻觉）
    3. 可能超出预期范围（费用失控）
    4. 用户需要保持控制感
    """

    print("=" * 60)
    print("为什么需要 Human-in-the-Loop")
    print("=" * 60)

    risks = [
        {
            "risk": "危险操作",
            "example": "Agent 可能误删数据库记录或发送错误邮件",
            "solution": "关键操作前需要人工审批",
        },
        {
            "risk": "幻觉与错误",
            "example": "Agent 可能编造不存在的工具结果",
            "solution": "人工验证关键输出",
        },
        {
            "risk": "费用失控",
            "example": "Agent 可能无限调用付费 API",
            "solution": "设置预算上限 + 超限审批",
        },
        {
            "risk": "用户控制感",
            "example": "用户不知道 Agent 在做什么",
            "solution": "实时展示进度 + 允许中断",
        },
    ]

    for r in risks:
        print(f"\n  ️ {r['risk']}")
        print(f"    示例: {r['example']}")
        print(f"    方案: {r['solution']}")
    print()


# ============================================================================
# 第二部分：干预级别定义
# ============================================================================

class InterventionLevel(Enum):
    """人工干预级别"""
    NONE = "none"               # 无需干预，完全自主
    NOTIFY = "notify"           # 仅通知，不阻塞
    APPROVE = "approve"         # 需要审批才能继续
    MANUAL = "manual"           # 需要人工执行


class ToolRiskLevel(Enum):
    """工具风险等级"""
    LOW = "low"                 # 低风险（查询类）
    MEDIUM = "medium"           # 中风险（创建类）
    HIGH = "high"               # 高风险（删除/发送类）


class ApprovalRequest(BaseModel):
    """审批请求"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str = Field(description="工具名称")
    arguments: dict = Field(description="工具参数")
    risk_level: ToolRiskLevel = Field(description="风险等级")
    reason: str = Field(description="需要审批的原因")
    status: str = Field(default="pending")  # pending/approved/rejected
    created_at: datetime = Field(default_factory=datetime.now)
    human_response: Optional[str] = Field(default=None)
    human_response_at: Optional[datetime] = Field(default=None)


# ============================================================================
# 第三部分：审批系统实现
# ============================================================================

class ApprovalManager:
    """
    审批管理器

    管理所有待审批的请求，支持：
    - 创建审批请求
    - 人工审批/拒绝
    - 超时自动处理
    - 审批历史
    """

    def __init__(self, auto_approve_low_risk: bool = True):
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self.auto_approve_low_risk = auto_approve_low_risk

    def request_approval(
        self,
        tool_name: str,
        arguments: dict,
        risk_level: ToolRiskLevel,
        reason: str = "",
    ) -> ApprovalRequest:
        """
        创建审批请求

        Returns:
            ApprovalRequest 对象
        """
        # 低风险且设置了自动审批
        if risk_level == ToolRiskLevel.LOW and self.auto_approve_low_risk:
            request = ApprovalRequest(
                tool_name=tool_name,
                arguments=arguments,
                risk_level=risk_level,
                reason=reason or "低风险操作，自动审批",
                status="approved",
                human_response="auto_approved",
                human_response_at=datetime.now(),
            )
            self._history.append(request)
            print(f"  ✅ 自动审批: {tool_name} (低风险)")
            return request

        # 需要人工审批
        request = ApprovalRequest(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level,
            reason=reason,
        )
        self._pending[request.id] = request
        print(f"  ⏳ 等待审批: {tool_name} (ID: {request.id})")
        return request

    def approve(self, request_id: str, response: str = "approved") -> bool:
        """审批通过"""
        if request_id in self._pending:
            request = self._pending.pop(request_id)
            request.status = "approved"
            request.human_response = response
            request.human_response_at = datetime.now()
            self._history.append(request)
            print(f"  ✅ 审批通过: {request.tool_name}")
            return True
        return False

    def reject(self, request_id: str, reason: str = "rejected") -> bool:
        """拒绝"""
        if request_id in self._pending:
            request = self._pending.pop(request_id)
            request.status = "rejected"
            request.human_response = reason
            request.human_response_at = datetime.now()
            self._history.append(request)
            print(f"  ❌ 审批拒绝: {request.tool_name}")
            return True
        return False

    def get_pending(self) -> list[ApprovalRequest]:
        """获取待审批列表"""
        return list(self._pending.values())

    def is_approved(self, request_id: str) -> bool:
        """检查是否已审批"""
        for req in self._history:
            if req.id == request_id:
                return req.status == "approved"
        return False


# ============================================================================
# 第四部分：带审批的 Agent
# ============================================================================

class HumanInTheLoopAgent:
    """
    带人工干预的 Agent

    在关键操作前暂停，等待人工审批。
    支持中断和恢复。
    """

    # 工具风险映射
    TOOL_RISK_MAP = {
        "web_search": ToolRiskLevel.LOW,
        "calculator": ToolRiskLevel.LOW,
        "get_weather": ToolRiskLevel.LOW,
        "send_email": ToolRiskLevel.HIGH,
        "delete_data": ToolRiskLevel.HIGH,
        "create_record": ToolRiskLevel.MEDIUM,
    }

    def __init__(self):
        self.approval_manager = ApprovalManager()
        self.interrupted = False

    async def process_with_approval(self, user_input: str) -> str:
        """
        处理用户输入（带审批）

        流程：
        1. 分析用户意图
        2. 决定需要调用的工具
        3. 检查工具风险等级
        4. 高风险工具需要审批
        5. 审批通过后执行
        """
        print(f"\n{'='*50}")
        print(f"用户输入: {user_input}")
        print(f"{'='*50}")

        # 模拟工具调用决策
        tool_name = self._decide_tool(user_input)
        arguments = self._build_arguments(user_input)
        risk_level = self.TOOL_RISK_MAP.get(tool_name, ToolRiskLevel.MEDIUM)

        print(f"\n  决定调用工具: {tool_name}")
        print(f"  风险等级: {risk_level.value}")

        # 请求审批
        request = self.approval_manager.request_approval(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level,
            reason=f"用户请求: {user_input[:30]}",
        )

        # 检查审批状态
        if request.status == "approved":
            # 自动审批或直接通过
            result = self._execute_tool(tool_name, arguments)
            print(f"\n  工具结果: {result}")
            return f"已完成: {result}"
        else:
            # 等待人工审批（模拟）
            print(f"\n  ⏳ 等待人工审批...")
            # 在实际应用中，这里会暂停并等待用户操作
            # 模拟审批通过
            self.approval_manager.approve(request.id)
            result = self._execute_tool(tool_name, arguments)
            print(f"\n  工具结果: {result}")
            return f"已完成: {result}"

    def _decide_tool(self, user_input: str) -> str:
        """决定工具"""
        if "邮件" in user_input or "发送" in user_input:
            return "send_email"
        elif "删除" in user_input:
            return "delete_data"
        elif "天气" in user_input:
            return "get_weather"
        return "web_search"

    def _build_arguments(self, user_input: str) -> dict:
        """构建参数"""
        return {"query": user_input}

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具"""
        return f"{tool_name} 执行成功"


# ============================================================================
# 第五部分：中断与恢复机制
# ============================================================================

class InterruptibleAgent:
    """
    可中断的 Agent

    支持：
    - 用户主动中断
    - 中断后恢复
    - 保存中间状态
    """

    def __init__(self):
        self._interrupted = False
        self._checkpoint: Optional[dict] = None

    async def run_with_interrupt(self, task: str) -> str:
        """运行任务（支持中断）"""
        print(f"\n  开始任务: {task}")

        steps = ["分析需求", "收集信息", "处理数据", "生成结果"]

        for i, step in enumerate(steps):
            if self._interrupted:
                print(f"\n  ️ 任务在 '{step}' 步骤被中断")
                self._checkpoint = {
                    "task": task,
                    "completed_steps": steps[:i],
                    "remaining_steps": steps[i:],
                }
                return "任务已中断"

            print(f"  步骤 {i+1}/{len(steps)}: {step}")
            await asyncio.sleep(0.2)  # 模拟工作

        print(f"\n  ✅ 任务完成: {task}")
        return f"完成: {task}"

    def interrupt(self):
        """中断任务"""
        self._interrupted = True
        print("\n  🛑 收到中断信号")

    def resume(self) -> Optional[str]:
        """恢复任务"""
        if self._checkpoint:
            print(f"\n  恢复任务: {self._checkpoint['task']}")
            print(f"  已完成: {self._checkpoint['completed_steps']}")
            print(f"  待完成: {self._checkpoint['remaining_steps']}")
            self._interrupted = False
            self._checkpoint = None
            return "任务已恢复"
        return None


# ============================================================================
# 第六部分：演示
# ============================================================================

async def demonstrate_approval_flow():
    """演示审批流程"""
    print("=" * 60)
    print("审批流程演示")
    print("=" * 60)

    agent = HumanInTheLoopAgent()

    # 1. 低风险操作（自动审批）
    print("\n── 低风险操作 ──")
    result = await agent.process_with_approval("北京今天天气怎么样？")
    print(f"  结果: {result}")

    # 2. 高风险操作（需要审批）
    print("\n── 高风险操作 ──")
    result = await agent.process_with_approval("给用户发送一封邮件")
    print(f"  结果: {result}")

    # 3. 查看审批历史
    print(f"\n── 审批历史 ─")
    for req in agent.approval_manager._history:
        print(f"  [{req.status}] {req.tool_name}: {req.reason[:20]}")


async def demonstrate_interrupt_resume():
    """演示中断与恢复"""
    print("\n" + "=" * 60)
    print("中断与恢复演示")
    print("=" * 60)

    agent = InterruptibleAgent()

    # 1. 正常运行
    print("\n── 正常运行 ─")
    await agent.run_with_interrupt("生成报告")

    # 2. 中断
    print("\n── 中断任务 ──")
    agent._interrupted = False  # 重置
    # 模拟在步骤2中断
    task = asyncio.create_task(agent.run_with_interrupt("分析数据"))
    await asyncio.sleep(0.5)  # 等待步骤1完成
    agent.interrupt()
    result = await task
    print(f"  结果: {result}")

    # 3. 恢复
    print("\n── 恢复任务 ──")
    resume_result = agent.resume()
    if resume_result:
        await agent.run_with_interrupt("分析数据（恢复）")


def demonstrate_intervention_levels():
    """演示不同干预级别"""
    print("\n" + "=" * 60)
    print("干预级别演示")
    print("=" * 60)

    levels = [
        {
            "level": "NONE（无干预）",
            "scenario": "查询天气、简单计算",
            "behavior": "Agent 自主执行，无需人工介入",
        },
        {
            "level": "NOTIFY（仅通知）",
            "scenario": "创建新记录、生成报告",
            "behavior": "Agent 执行后通知用户，用户可查看但不阻塞",
        },
        {
            "level": "APPROVE（需审批）",
            "scenario": "发送邮件、修改数据",
            "behavior": "Agent 暂停，等待用户审批后继续",
        },
        {
            "level": "MANUAL（人工执行）",
            "scenario": "银行转账、删除重要数据",
            "behavior": "Agent 生成操作指令，由用户手动执行",
        },
    ]

    for l in levels:
        print(f"\n🔷 {l['level']}")
        print(f"   场景: {l['scenario']}")
        print(f"   行为: {l['behavior']}")
    print()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "" * 30)
    print("第十五课（补充）：Human-in-the-Loop（人机协作）")
    print("👤" * 30 + "\n")

    # 1. 为什么需要
    why_human_in_the_loop()

    # 2. 干预级别
    demonstrate_intervention_levels()

    # 3. 审批流程
    asyncio.run(demonstrate_approval_flow())

    # 4. 中断与恢复
    asyncio.run(demonstrate_interrupt_resume())

    print("=" * 60)
    print("✅ 第十五课学习完成！")
    print("=" * 60)
    print("""
 本课要点总结：
1. Human-in-the-Loop 确保 Agent 安全可控
2. 根据工具风险等级设置不同的干预级别
3. 审批系统管理所有待审批请求
4. 支持任务中断和恢复
5. 关键操作（删除、发送、转账）必须人工审批
    """)
