"""
主题：第一次调用 Claude API —— Hello Claude!

学习目标：
  1. 掌握如何通过 .env 文件安全管理 API Key
  2. 学会初始化 Anthropic 客户端
  3. 理解 messages.create() 的三个核心参数：model / max_tokens / messages
  4. 学会从响应中提取文本内容
  5. 了解 API 的 token 使用量统计
  6. 看懂完整的响应对象结构（为后续课程打基础）

前置知识：
  - Python 基础（函数、变量、print、f-string）
  - 已安装依赖：pip install anthropic python-dotenv
  - 已在项目根目录创建 .env 文件，内容为：
      ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

课程顺序：这是 01_basics 模块的第一个文件，建议按序学习。
"""

# ── 0. Windows 控制台编码修复 ──────────────────────────────────────────────────
# Windows 默认控制台编码（GBK）无法显示部分 Unicode 字符（如 emoji）
# 将 stdout/stderr 强制设为 UTF-8，确保中文和 emoji 都能正常打印
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── 1. 导入所需模块 ────────────────────────────────────────────────────────────
import os          # 标准库：用于读取环境变量

import anthropic   # Anthropic 官方 SDK，提供与 Claude API 交互的所有工具

# python-dotenv：将 .env 文件中的键值对加载到操作系统环境变量中
# 这样 os.getenv() 就能读到我们配置的 ANTHROPIC_API_KEY
from dotenv import load_dotenv


# ── 2. 加载环境变量 ────────────────────────────────────────────────────────────
# load_dotenv() 会自动在当前目录及上级目录查找 .env 文件并加载
# 最佳实践：API Key 永远不要硬编码在代码里，.env 文件也不要提交到 Git
load_dotenv()


# ── 3. 初始化 Anthropic 客户端 ─────────────────────────────────────────────────
# anthropic.Anthropic() 会自动读取环境变量 ANTHROPIC_API_KEY
# 你也可以显式传入：anthropic.Anthropic(api_key="sk-ant-xxx")，但不推荐
# 客户端初始化后可以复用，不需要每次调用都重新创建
client = anthropic.Anthropic()


def main():
    print("=*" * 30)
    print("Hello Claude! —— 第一次 API 调用")
    print("=#" * 30)

    # ── 4. 发送第一条消息 ──────────────────────────────────────────────────────
    # messages.create() 是最核心的 API，三个必填参数说明如下：
    #
    # model：要调用的 Claude 模型版本
    #   - 不同模型在速度、能力、价格上有差异
    #   - 推荐使用最新的稳定版本，此处使用 claude-sonnet-4-6
    #
    # max_tokens：允许模型输出的最大 token 数量
    #   - token 约等于 0.75 个英文单词 / 0.5 个中文字符（粗略估算）
    #   - 设置合理的上限可以控制响应长度和 API 费用
    #   - 这里设为 1024，足够一次简短对话使用
    #
    # messages：对话消息列表，每条消息是一个字典，包含：
    #   - "role"：消息角色，"user" 表示用户发送的消息
    #             （另一个角色是 "assistant"，用于多轮对话，后续课程会讲到）
    #   - "content"：消息的文字内容
    print("\n正在向 Claude 发送问候消息...")

    response = client.messages.create(
        model="ppio/pa/claude-sonnet-4-6",  # 使用 claude-sonnet-4-6 模型（通过代理前缀 ppio/pa/）
        max_tokens=1024,                # 最多生成 1024 个 token
        messages=[
            {
                "role": "user",
                "content": "你好，Claude！请用一句话介绍你自己，并告诉我今天适合做什么事情。"
            }
        ]
    )

    # ── 5. 解析并打印响应文本 ──────────────────────────────────────────────────
    # response.content 是一个列表，因为 Claude 可能返回多个内容块
    # （例如：文本 + 工具调用结果，后续课程会讲 tool_use）
    # 对于普通文本回复，直接取第一个元素 [0]，访问其 .text 属性即可
    reply_text = response.content[0].text

    print("\n── Claude 的回复 ──────────────────────────────────────")
    print(reply_text)

    # ── 6. 打印 token 使用量统计 ───────────────────────────────────────────────
    # response.usage 包含本次 API 调用的 token 消耗，用于费用估算和调试
    # input_tokens：你发送的消息（含系统提示）消耗的 token 数
    # output_tokens：Claude 生成的回复消耗的 token 数
    # 计费 = input_tokens × 输入价格 + output_tokens × 输出价格
    print("\n── Token 使用量 ────────────────────────────────────────")
    print(f"  输入 tokens：{response.usage.input_tokens}")
    print(f"  输出 tokens：{response.usage.output_tokens}")
    print(f"  合计 tokens：{response.usage.input_tokens + response.usage.output_tokens}")

    # ── 7. 打印完整响应结构（供学习参考）─────────────────────────────────────
    # 初学阶段建议打印完整对象，帮助理解 API 返回了哪些字段
    # Message 对象的主要字段：
    #   id          - 本次请求的唯一 ID，调试时可用于追踪问题
    #   type        - 固定为 "message"
    #   role        - 固定为 "assistant"，表示这是 Claude 的回复
    #   content     - 内容块列表，通常包含一个 TextBlock
    #   model       - 实际使用的模型名称（与请求参数一致）
    #   stop_reason - 停止原因："end_turn" 表示正常结束；
    #                 其他值："max_tokens"（达到长度限制）、"stop_sequence" 等
    #   usage       - token 使用量（见第6步）
    print("\n── 完整响应对象结构 ─────────────────────────────────────")
    print(f"  response.id          = {response.id}")
    print(f"  response.type        = {response.type}")
    print(f"  response.role        = {response.role}")
    print(f"  response.model       = {response.model}")
    print(f"  response.stop_reason = {response.stop_reason}")
    print(f"  response.content     = {response.content}")

    print("\n验证通过：第一次 API 调用成功！")
    print("=" * 60)


if __name__ == "__main__":
    # 只有直接运行此文件时才执行 main()
    # 如果被其他模块 import，则不会自动执行（这是 Python 的标准实践）
    main()
