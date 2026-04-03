import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
主题：Tools & Agents —— 让 AI 能调用工具，自主完成任务

学习目标：
  1. 理解 Tool 的概念：AI 可以调用的函数
  2. 掌握 @tool 装饰器定义工具
  3. 理解 ReAct 推理模式（思考→行动→观察→循环）
  4. 掌握 create_react_agent + AgentExecutor
  5. 学会观察 Agent 的推理过程（verbose=True）

核心概念：
  Tool = 有名字、有描述、可被 LLM 调用的 Python 函数
  Agent = 能自主决定调用哪些工具、以什么顺序、来完成目标的 LLM

  ReAct = Reason + Act
  循环：思考（要做什么）→ 行动（调用工具）→ 观察（工具结果）→ 继续思考

前置知识：已完成 03_lcel_chains.py
"""

import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

load_dotenv()


# ============================================================
# 工具定义（全局）
# ============================================================

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式并返回结果。
    支持基本四则运算：加减乘除，以及括号组合。
    示例：'123 * 456'，'(10 + 5) * 3'，'100 / 4'
    """
    try:
        # 安全限制：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return f"错误：表达式包含不允许的字符，只支持 0-9 和 + - * / ( ) 空格"
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def word_count(text: str) -> str:
    """
    统计文本中的词语数量。
    对中文文本按字符计数，对英文文本按空格分词计数。
    示例：输入 'hello world' 返回词数 2；输入 'LangChain 框架' 返回词数 2
    """
    # 简单分词：按空格分割，过滤空字符串
    words = [w for w in text.split() if w]
    if words:
        word_num = len(words)
    else:
        # 纯中文无空格时，按字符数统计
        word_num = len(text.strip())

    return f"文本「{text[:30]}{'...' if len(text) > 30 else ''}」共 {word_num} 个词（按空格分词）"


@tool
def get_current_date() -> str:
    """
    获取今天的日期，返回格式为 YYYY年MM月DD日（星期X）。
    用于回答"今天是几号""今天是星期几"等问题。
    """
    today = datetime.date.today()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[today.weekday()]
    return f"今天是 {today.year}年{today.month:02d}月{today.day:02d}日，{weekday}"


# 工具列表（Agent 可调用的全部工具）
TOOLS = [calculate, word_count, get_current_date]


# ============================================================
# 构建 ReAct Prompt（带 fallback）
# ============================================================

def get_react_prompt():
    """
    获取 ReAct Agent 所需的 Prompt 模板。

    优先尝试从 LangChain Hub 拉取标准 ReAct Prompt，
    若网络不通则使用内置的 fallback 版本。

    ReAct Prompt 的核心是规定 Agent 的输出格式：
      Thought → Action → Action Input → Observation → ... → Final Answer
    """
    try:
        from langchain import hub
        prompt = hub.pull("hwchase17/react")
        print("  已从 LangChain Hub 加载标准 ReAct Prompt")
        return prompt
    except Exception as e:
        print(f"  Hub 拉取失败（{e}），使用内置 fallback Prompt")
        # 内置的 ReAct Prompt 模板
        react_prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        )
        return react_prompt


# ============================================================
# Part 1: 定义工具并展示工具信息
# ============================================================

def part1_define_tools():
    """
    @tool 装饰器把一个普通 Python 函数转换为 LangChain Tool 对象。

    LLM 通过工具的名称和描述来决定"什么时候调用哪个工具"。
    因此，工具的 docstring（描述）要清晰准确，这直接影响 Agent 的判断质量。

    Tool 对象的关键属性：
      name        = 函数名（LLM 用这个名字发起调用）
      description = docstring（LLM 据此判断工具用途）
      args_schema = 输入参数的类型定义（自动从函数签名生成）
    """
    print("=" * 60)
    print("Part 1: 定义工具 —— @tool 装饰器")
    print("=" * 60)

    print(f"共定义 {len(TOOLS)} 个工具：")
    print()

    for t in TOOLS:
        print(f"  工具名称：{t.name}")
        # description 来自 docstring，LLM 用它判断何时调用此工具
        desc_preview = t.description.strip().split("\n")[0]
        print(f"  工具描述：{desc_preview}")
        print()


# ============================================================
# Part 2: 创建 ReAct Agent 并执行单步任务
# ============================================================

def part2_create_and_run_agent():
    """
    ReAct Agent 的工作原理：

    1. LLM 接收用户问题和工具描述
    2. LLM 思考（Thought）：需要做什么？
    3. LLM 行动（Action）：选择工具并提供输入
    4. 系统执行工具，返回观察结果（Observation）
    5. LLM 根据观察结果继续思考，或给出最终答案（Final Answer）

    verbose=True 会打印完整的推理过程，非常有助于学习理解 Agent 行为。
    """
    print("=" * 60)
    print("Part 2: ReAct Agent —— 执行多工具任务")
    print("=" * 60)

    llm = ChatAnthropic(
        model="ppio/pa/claude-sonnet-4-6",
        max_tokens=1024,
    )

    print("加载 ReAct Prompt...")
    prompt = get_react_prompt()
    print()

    # 创建 ReAct Agent
    # create_react_agent 把 llm + tools + prompt 组合成 Agent
    agent = create_react_agent(llm, TOOLS, prompt)

    # AgentExecutor 负责实际运行 Agent，处理工具调用循环
    # verbose=True：打印每一步的 Thought/Action/Observation
    # max_iterations：防止无限循环（默认 15）
    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,  # 遇到解析错误时自动重试
    )

    # 任务：需要用到 get_current_date 和 calculate 两个工具
    task = "今天是几月几日？另外计算 123 * 456 等于多少？"
    print(f"任务：{task}")
    print()
    print("【Agent 推理过程（verbose=True）】")
    print("-" * 40)

    result = agent_executor.invoke({"input": task})

    print("-" * 40)
    print()
    print("【最终回答】")
    print(result["output"])
    print()

    return agent_executor


# ============================================================
# Part 3: 多工具组合任务
# ============================================================

def part3_multi_tool_task(agent_executor):
    """
    观察 Agent 如何分解复杂任务并依次调用多个工具。

    任务包含两个子目标：
      1. 统计指定文本的词数 → 需要 word_count 工具
      2. 获取今天的日期    → 需要 get_current_date 工具

    Agent 会自主规划执行顺序，不需要人工指定调用哪个工具。
    这就是 Agent 与普通函数调用的本质区别——LLM 自主决策。
    """
    print("=" * 60)
    print("Part 3: 多工具组合任务")
    print("=" * 60)

    task = "帮我计算这段文字有多少个词：'LangChain 是一个强大的 AI 应用开发框架'，然后告诉我今天的日期"
    print(f"任务：{task}")
    print()
    print("【Agent 推理过程】")
    print("-" * 40)

    result = agent_executor.invoke({"input": task})

    print("-" * 40)
    print()
    print("【最终回答】")
    print(result["output"])
    print()


# ============================================================
# main
# ============================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     11_tools_agents.py —— Tools & ReAct Agent 演示       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    part1_define_tools()
    agent_executor = part2_create_and_run_agent()
    part3_multi_tool_task(agent_executor)

    print("=" * 60)
    print("Tools & Agents 演示完毕！")
    print()
    print("关键要点：")
    print("  • @tool 把普通函数转为 LangChain 可调用的工具")
    print("  • Tool 的 docstring 是 LLM 判断何时调用的依据，要写清楚")
    print("  • ReAct = 推理（Reason）+ 行动（Act），循环直至完成目标")
    print("  • create_react_agent + AgentExecutor 是标准 ReAct 实现")
    print("  • verbose=True 打印完整推理链路，方便调试和学习")
    print("=" * 60)


if __name__ == "__main__":
    main()
