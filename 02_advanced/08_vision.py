"""
Vision / 多模态（图片理解）完整演示
=====================================
本文件演示 Claude API 的视觉能力，展示如何向 Claude 发送图片并让其理解图片内容。

核心知识点：
1. 用纯 Python 标准库（zlib + struct）生成测试 PNG 图片（无需 Pillow）
2. 用 base64.b64encode() 将图片编码为 base64 字符串
3. image content block 格式（type/source/media_type/data）
4. content 列表中混合 image block 和 text block
5. 让 Claude 描述图片、分析颜色/形状等实际场景
"""

import sys
import os
import base64
import zlib
import struct

# 解决 Windows 中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "ppio/pa/claude-sonnet-4-6"

# ============================================================
# 第一部分：生成测试 PNG 图片（不依赖任何第三方库）
# ============================================================

def create_minimal_png(width: int = 16, height: int = 16,
                       color: tuple = (255, 0, 0)) -> bytes:
    """
    用 Python 标准库生成一个纯色方块 PNG，返回 bytes。

    PNG 文件结构：
      签名（8 字节） + IHDR chunk + IDAT chunk + IEND chunk

    每个 chunk 的结构：
      4B 数据长度 | 4B chunk 类型 | N字节数据 | 4B CRC32
    """
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        """构造一个 PNG chunk"""
        chunk_len = len(data)
        chunk_data = chunk_type + data
        crc = zlib.crc32(chunk_data) & 0xFFFFFFFF
        return struct.pack('>I', chunk_len) + chunk_data + struct.pack('>I', crc)

    # PNG 魔数签名
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR：图像头部信息
    #   width(4B) height(4B) bit_depth(1B) color_type(2=RGB)(1B)
    #   compression(1B) filter(1B) interlace(1B)
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b'IHDR', ihdr_data)

    # IDAT：图像像素数据（压缩）
    # 每行以 filter-type 字节（0=None）开头，后跟 RGB 像素
    r, g, b = color
    raw_data = b''
    for _ in range(height):
        raw_data += b'\x00'                   # filter byte
        for _ in range(width):
            raw_data += bytes([r, g, b])       # RGB 像素
    compressed = zlib.compress(raw_data)
    idat = make_chunk(b'IDAT', compressed)

    # IEND：文件结束标记
    iend = make_chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


def generate_test_image(color: tuple = (255, 0, 0),
                        label: str = "红色") -> str:
    """
    生成测试用 PNG 图片，返回 base64 字符串。

    参数：
        color: RGB 颜色元组，默认红色 (255, 0, 0)
        label: 颜色名称（用于打印提示）

    返回：
        base64 编码的 PNG 字符串（不含换行符）
    """
    png_bytes = create_minimal_png(width=32, height=32, color=color)
    b64_str = base64.b64encode(png_bytes).decode('utf-8')
    print(f"  [生成测试图片] {label}方块 PNG，"
          f"大小={len(png_bytes)} bytes，"
          f"base64长度={len(b64_str)} chars")
    return b64_str


# ============================================================
# 第二部分：构造 image content block
# ============================================================

def build_image_block(b64_data: str, media_type: str = "image/png") -> dict:
    """
    构造符合 Claude API 规范的 image content block。

    格式：
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",   # 支持 png/jpeg/gif/webp
            "data": "<base64字符串>"
        }
    }
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64_data
        }
    }


def build_text_block(text: str) -> dict:
    """构造文本 content block"""
    return {
        "type": "text",
        "text": text
    }


# ============================================================
# 演示函数
# ============================================================

def demo_image_description():
    """
    演示 1：图片描述
    让 Claude 描述一张纯色方块图片，验证基本视觉能力。
    """
    print("\n" + "=" * 60)
    print("演示 1：图片描述（Image Description）")
    print("=" * 60)

    # 生成一张红色方块
    b64_image = generate_test_image(color=(255, 0, 0), label="红色")

    # 构造包含图片的消息
    # content 是一个列表，可以同时包含 image block 和 text block
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block(b64_image),           # 图片 block
                    build_text_block("请描述这张图片的内容。")  # 文字 block
                ]
            }
        ]
    )

    print(f"\n[Claude 回复]")
    print(response.content[0].text)
    print(f"\n[Token 用量] 输入={response.usage.input_tokens}, "
          f"输出={response.usage.output_tokens}")


def demo_text_and_image():
    """
    演示 2：文字 + 图片混合消息
    content 列表中先放文字说明，再放图片，展示混合排列方式。
    """
    print("\n" + "=" * 60)
    print("演示 2：文字 + 图片混合消息（Mixed Content）")
    print("=" * 60)

    # 生成一张蓝色方块
    b64_image = generate_test_image(color=(0, 100, 255), label="蓝色")

    # content 列表中文字在前、图片在后（顺序可自由调整）
    content = [
        build_text_block("我给你发了一张图片，"),  # 先放文字
        build_image_block(b64_image),              # 再放图片
        build_text_block("请告诉我：图中主要颜色是什么？形状是什么？")  # 追加问题
    ]

    print(f"\n[消息结构] content 包含 {len(content)} 个 block：")
    for i, block in enumerate(content):
        print(f"  [{i}] type={block['type']}", end="")
        if block['type'] == 'text':
            print(f"  → \"{block['text']}\"")
        else:
            print(f"  → <base64 image, {len(block['source']['data'])} chars>")

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": content}]
    )

    print(f"\n[Claude 回复]")
    print(response.content[0].text)


def demo_image_analysis():
    """
    演示 3：图片分析（颜色 / 形状识别）
    发送多张不同颜色的方块，让 Claude 逐一分析并对比。
    """
    print("\n" + "=" * 60)
    print("演示 3：多图对比分析（Multi-Image Analysis）")
    print("=" * 60)

    # 生成三张不同颜色的方块
    colors = [
        ((255, 0,   0),   "红色"),
        ((0,   255, 0),   "绿色"),
        ((0,   0,   255), "蓝色"),
    ]

    content = [build_text_block("下面有三张图片，请分别说明每张图片的颜色，并按顺序列出：\n")]

    for color_rgb, label in colors:
        b64 = generate_test_image(color=color_rgb, label=label)
        content.append(build_text_block(f"图片（{label}）："))
        content.append(build_image_block(b64))

    content.append(build_text_block("\n请逐一描述每张图的颜色。"))

    print(f"\n[消息结构] 共 {len(content)} 个 content block（含 3 张图片）")

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": content}]
    )

    print(f"\n[Claude 回复]")
    print(response.content[0].text)
    print(f"\n[Token 用量] 输入={response.usage.input_tokens}, "
          f"输出={response.usage.output_tokens}")


def demo_base64_from_file(file_path: str | None = None):
    """
    演示 4：从本地文件读取图片并发送（可选演示）
    若未提供文件路径，则跳过此演示。

    展示知识点：读取本地文件 → base64 编码 → 构造 image block
    """
    print("\n" + "=" * 60)
    print("演示 4：从本地文件读取图片（Local File Base64）")
    print("=" * 60)

    if file_path is None or not os.path.exists(file_path):
        print("[跳过] 未提供有效的本地图片路径。")
        print("  用法示例：demo_base64_from_file('path/to/image.png')")
        print("\n  核心代码逻辑（供参考）：")
        print("  ┌─────────────────────────────────────────────┐")
        print("  │ with open('image.png', 'rb') as f:           │")
        print("  │     raw = f.read()                           │")
        print("  │ b64 = base64.b64encode(raw).decode('utf-8') │")
        print("  │ block = build_image_block(b64, 'image/png') │")
        print("  └─────────────────────────────────────────────┘")
        return

    # 根据扩展名判断 media_type
    ext = os.path.splitext(file_path)[1].lower()
    media_map = {'.png': 'image/png', '.jpg': 'image/jpeg',
                 '.jpeg': 'image/jpeg', '.gif': 'image/gif',
                 '.webp': 'image/webp'}
    media_type = media_map.get(ext, 'image/png')

    # 读取文件 → base64 编码
    with open(file_path, 'rb') as f:
        raw_bytes = f.read()
    b64_data = base64.b64encode(raw_bytes).decode('utf-8')
    print(f"  已读取文件：{file_path}，{len(raw_bytes)} bytes → base64 {len(b64_data)} chars")

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block(b64_data, media_type),
                    build_text_block("请描述这张图片的内容。")
                ]
            }
        ]
    )

    print(f"\n[Claude 回复]")
    print(response.content[0].text)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Claude Vision / 多模态图片理解 演示")
    print("=" * 60)
    print(f"使用模型：{MODEL}")
    print()
    print("知识点总览：")
    print("  1. 用 struct + zlib 生成最小 PNG（无需 Pillow）")
    print("  2. base64.b64encode() 编码图片数据")
    print("  3. image content block 格式（type/source/media_type/data）")
    print("  4. content 列表中混合 image block 与 text block")
    print("  5. 实际场景：图片描述、颜色识别、多图对比")

    # 运行各演示
    demo_image_description()
    demo_text_and_image()
    demo_image_analysis()
    demo_base64_from_file()   # 无本地文件时自动跳过

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
