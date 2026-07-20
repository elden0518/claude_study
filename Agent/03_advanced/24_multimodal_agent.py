"""
==============================================================================
第二十四课（补充）：多模态 Agent
==============================================================================

【为什么需要单独一课？】
现有课程只涉及文本处理，没有讲解如何处理图片、音频等多模态输入。
生产级 Agent 需要能够理解和处理多种类型的数据。

【学习目标】
- 理解多模态 Agent 的架构
- 掌握视觉理解（图片描述、OCR）
- 掌握音频处理（语音转文字、情感分析）
- 学会构建多模态处理管道
- 理解多模态融合策略

【核心概念】
- Vision Understanding（视觉理解）
- Audio Processing（音频处理）
- Document Understanding（文档理解）
- Multimodal Fusion（多模态融合）

==============================================================================
"""

import base64
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ============================================================================
# 第一部分：多模态内容类型
# ============================================================================

class MediaType(Enum):
    """媒体类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


@dataclass
class MediaContent:
    """多模态内容"""
    media_type: MediaType
    content: Union[str, bytes]
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            if isinstance(self.content, str):
                self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            else:
                self.content_hash = hashlib.md5(self.content).hexdigest()[:8]


@dataclass
class MultimodalInput:
    """多模态输入"""
    contents: List[MediaContent] = field(default_factory=list)
    context: str = ""

    def add_text(self, text: str, metadata: Dict = None):
        self.contents.append(MediaContent(
            media_type=MediaType.TEXT,
            content=text,
            metadata=metadata or {}
        ))
        return self

    def add_image(self, image_data: Union[str, bytes], metadata: Dict = None):
        self.contents.append(MediaContent(
            media_type=MediaType.IMAGE,
            content=image_data,
            metadata=metadata or {}
        ))
        return self

    def add_audio(self, audio_data: Union[str, bytes], metadata: Dict = None):
        self.contents.append(MediaContent(
            media_type=MediaType.AUDIO,
            content=audio_data,
            metadata=metadata or {}
        ))
        return self


# ============================================================================
# 第二部分：视觉处理器
# ============================================================================

class VisionProcessor:
    """
    视觉处理器

    【功能】
    - 图片描述生成
    - OCR 文字识别
    - 图表分析
    - 物体检测
    """

    def __init__(self):
        self._processed_count = 0

    def describe_image(self, image_content: MediaContent) -> Dict[str, Any]:
        """
        图片描述

        【原理】
        使用视觉模型分析图片内容，生成自然语言描述
        """
        print(f"    [Vision] 分析图片 (hash: {image_content.content_hash})...")
        self._processed_count += 1

        # 模拟分析结果
        return {
            "description": "图片展示了一个数据可视化图表，包含柱状图和折线图",
            "objects": ["图表", "数据", "坐标轴", "图例"],
            "text_found": ["Q1", "Q2", "Q3", "Q4", "销售额"],
            "dominant_colors": ["蓝色", "白色", "灰色"],
            "confidence": 0.92,
            "processing_time_ms": 150
        }

    def extract_text(self, image_content: MediaContent) -> Dict[str, Any]:
        """
        OCR 文字提取

        【原理】
        识别图片中的文字并返回结构化结果
        """
        print(f"    [OCR] 提取文字 (hash: {image_content.content_hash})...")
        self._processed_count += 1

        return {
            "text": "销售额报告\nQ1: 100万\nQ2: 150万\nQ3: 120万\nQ4: 180万",
            "language": "zh",
            "confidence": 0.95,
            "regions": [
                {"text": "销售额报告", "bbox": [10, 10, 200, 40]},
                {"text": "Q1: 100万", "bbox": [10, 50, 150, 70]},
            ]
        }

    def analyze_chart(self, image_content: MediaContent) -> Dict[str, Any]:
        """
        图表分析

        【原理】
        专门分析图表类型图片，提取数据趋势
        """
        print(f"    [Chart] 分析图表 (hash: {image_content.content_hash})...")
        self._processed_count += 1

        return {
            "chart_type": "bar_chart",
            "title": "季度销售额",
            "data_points": [
                {"label": "Q1", "value": 100},
                {"label": "Q2", "value": 150},
                {"label": "Q3", "value": 120},
                {"label": "Q4", "value": 180},
            ],
            "trend": "增长趋势",
            "insights": ["Q4 销售额最高", "Q2 环比增长 50%"]
        }

    def get_stats(self) -> Dict[str, int]:
        return {"processed_count": self._processed_count}


# ============================================================================
# 第三部分：音频处理器
# ============================================================================

class AudioProcessor:
    """
    音频处理器

    【功能】
    - 语音转文字（ASR）
    - 说话人识别
    - 情感分析
    - 关键词提取
    """

    def __init__(self):
        self._processed_count = 0

    def speech_to_text(self, audio_content: MediaContent) -> Dict[str, Any]:
        """
        语音转文字

        【原理】
        使用 ASR 模型将音频转换为文本
        """
        print(f"    [ASR] 语音转文字 (hash: {audio_content.content_hash})...")
        self._processed_count += 1

        return {
            "text": "你好，我想咨询一下关于产品退货的问题。我上周购买的商品有质量问题。",
            "language": "zh-CN",
            "duration_seconds": 8.5,
            "confidence": 0.94,
            "words_count": 32
        }

    def identify_speaker(self, audio_content: MediaContent) -> Dict[str, Any]:
        """
        说话人识别

        【原理】
        通过声纹特征识别说话人
        """
        print(f"    [Speaker] 识别说话人 (hash: {audio_content.content_hash})...")
        self._processed_count += 1

        return {
            "speakers": [
                {"id": "speaker_1", "name": "用户", "segments": [(0.0, 8.5)]},
            ],
            "total_speakers": 1,
            "confidence": 0.88
        }

    def analyze_sentiment(self, audio_content: MediaContent) -> Dict[str, Any]:
        """
        情感分析

        【原理】
        通过语音特征（语调、语速、音量）分析情感
        """
        print(f"    [Sentiment] 分析情感 (hash: {audio_content.content_hash})...")
        self._processed_count += 1

        return {
            "sentiment": "negative",
            "confidence": 0.78,
            "emotions": {
                "frustration": 0.6,
                "concern": 0.7,
                "anger": 0.3,
                "neutral": 0.4
            },
            "tone_indicators": {
                "pitch_variation": "high",
                "speech_rate": "fast",
                "volume": "loud"
            }
        }

    def extract_keywords(self, text: str) -> List[str]:
        """从转录文本提取关键词"""
        # 简化的关键词提取
        stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个"}
        words = [w for w in text.split() if w not in stop_words and len(w) > 1]
        return list(set(words))[:10]

    def get_stats(self) -> Dict[str, int]:
        return {"processed_count": self._processed_count}


# ============================================================================
# 第四部分：文档处理器
# ============================================================================

class DocumentProcessor:
    """
    文档处理器

    【功能】
    - PDF 解析
    - 表格提取
    - 结构化信息提取
    """

    def __init__(self):
        self._processed_count = 0

    def parse_document(self, doc_content: MediaContent) -> Dict[str, Any]:
        """
        解析文档

        【原理】
        提取文档的结构化信息
        """
        print(f"    [Doc] 解析文档 (hash: {doc_content.content_hash})...")
        self._processed_count += 1

        return {
            "page_count": 5,
            "title": "产品需求文档",
            "sections": ["背景", "目标", "功能需求", "非功能需求", "时间计划"],
            "tables_count": 3,
            "images_count": 2,
            "word_count": 2500
        }

    def extract_tables(self, doc_content: MediaContent) -> List[Dict]:
        """
        提取表格

        【原理】
        识别文档中的表格并转换为结构化数据
        """
        print(f"    [Table] 提取表格 (hash: {doc_content.content_hash})...")
        self._processed_count += 1

        return [
            {
                "table_id": 1,
                "headers": ["功能", "优先级", "预计工时"],
                "rows": [
                    ["用户登录", "P0", "2天"],
                    ["数据导入", "P1", "3天"],
                    ["报表生成", "P2", "5天"],
                ]
            }
        ]

    def extract_key_info(self, doc_content: MediaContent) -> Dict[str, Any]:
        """
        提取关键信息

        【原理】
        从文档中提取关键实体和信息
        """
        print(f"    [KeyInfo] 提取关键信息 (hash: {doc_content.content_hash})...")
        self._processed_count += 1

        return {
            "entities": [
                {"type": "DATE", "value": "2024年Q1"},
                {"type": "PERSON", "value": "张三"},
                {"type": "ORG", "value": "技术部"},
            ],
            "key_points": [
                "项目目标是提升用户体验",
                "预计 3 个月完成",
                "需要 5 人团队",
            ]
        }

    def get_stats(self) -> Dict[str, int]:
        return {"processed_count": self._processed_count}


# ============================================================================
# 第五部分：多模态 Agent
# ============================================================================

class MultimodalAgent:
    """
    多模态 Agent

    【功能】
    整合多种模态处理能力：
    - 视觉理解
    - 音频处理
    - 文档理解
    - 文本推理
    """

    def __init__(self):
        self.vision = VisionProcessor()
        self.audio = AudioProcessor()
        self.document = DocumentProcessor()

    def process(self, input_data: MultimodalInput) -> Dict[str, Any]:
        """
        处理多模态输入

        【流程】
        1. 分析输入类型
        2. 调用对应的处理器
        3. 融合多模态结果
        4. 生成最终回复
        """
        print(f"\n{'='*60}")
        print("多模态 Agent 处理")
        print(f"{'='*60}")
        print(f"  输入内容数: {len(input_data.contents)}")
        print(f"  上下文: {input_data.context}")

        results = {}

        for i, content in enumerate(input_data.contents):
            print(f"\n  [Content {i+1}] 类型: {content.media_type.value}")

            if content.media_type == MediaType.TEXT:
                results["text"] = {"content": content.content}

            elif content.media_type == MediaType.IMAGE:
                # 视觉处理
                vision_result = self.vision.describe_image(content)
                ocr_result = self.vision.extract_text(content)
                results["image"] = {
                    "description": vision_result["description"],
                    "text_extracted": ocr_result["text"],
                    "objects": vision_result["objects"]
                }

            elif content.media_type == MediaType.AUDIO:
                # 音频处理
                asr_result = self.audio.speech_to_text(content)
                sentiment_result = self.audio.analyze_sentiment(content)
                results["audio"] = {
                    "transcript": asr_result["text"],
                    "sentiment": sentiment_result["sentiment"],
                    "duration": asr_result["duration_seconds"]
                }

            elif content.media_type == MediaType.DOCUMENT:
                # 文档处理
                doc_result = self.document.parse_document(content)
                tables = self.document.extract_tables(content)
                key_info = self.document.extract_key_info(content)
                results["document"] = {
                    "summary": doc_result,
                    "tables": tables,
                    "key_info": key_info
                }

        # 融合结果
        final_response = self._fuse_results(results, input_data.context)
        return final_response

    def _fuse_results(self, results: Dict, context: str) -> Dict[str, Any]:
        """
        融合多模态结果

        【策略】
        将所有模态的处理结果整合为统一的回复
        """
        print(f"\n  [Fusion] 融合 {len(results)} 个模态的结果...")

        fused = {
            "context": context,
            "modalities_processed": list(results.keys()),
            "summary": self._generate_summary(results),
            "details": results
        }

        return fused

    def _generate_summary(self, results: Dict) -> str:
        """生成综合摘要"""
        parts = []

        if "text" in results:
            parts.append(f"文本内容: {results['text']['content'][:50]}...")

        if "image" in results:
            parts.append(f"图片描述: {results['image']['description']}")
            if results['image']['text_extracted']:
                parts.append(f"提取文字: {results['image']['text_extracted'][:30]}...")

        if "audio" in results:
            parts.append(f"语音内容: {results['audio']['transcript'][:30]}...")
            parts.append(f"情感倾向: {results['audio']['sentiment']}")

        if "document" in results:
            doc = results['document']
            parts.append(f"文档页数: {doc['summary']['page_count']}")
            parts.append(f"关键信息: {len(doc['key_info']['key_points'])} 条")

        return " | ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        return {
            "vision": self.vision.get_stats(),
            "audio": self.audio.get_stats(),
            "document": self.document.get_stats()
        }


# ============================================================================
# 第六部分：完整示例
# ============================================================================

def demo_multimodal_agent():
    """演示多模态 Agent"""

    print("=" * 60)
    print("多模态 Agent 完整演示")
    print("=" * 60)

    # 创建 Agent
    agent = MultimodalAgent()

    # 场景1: 图片分析
    print("\n" + "=" * 60)
    print("场景1: 图片分析")
    print("=" * 60)

    input1 = MultimodalInput(context="分析这张销售报告图片")
    input1.add_image(b"fake_image_data_12345")
    input1.add_text("请分析这张图片中的销售数据趋势")

    result1 = agent.process(input1)
    print(f"\n  综合结果:")
    print(f"    摘要: {result1['summary']}")
    print(f"    处理模态: {result1['modalities_processed']}")

    # 场景2: 语音客服
    print("\n" + "=" * 60)
    print("场景2: 语音客服")
    print("=" * 60)

    input2 = MultimodalInput(context="处理客户语音咨询")
    input2.add_audio(b"fake_audio_data_67890")

    result2 = agent.process(input2)
    print(f"\n  综合结果:")
    print(f"    摘要: {result2['summary']}")

    # 场景3: 文档理解
    print("\n" + "=" * 60)
    print("场景3: 文档理解")
    print("=" * 60)

    input3 = MultimodalInput(context="分析产品需求文档")
    input3.contents.append(MediaContent(
        media_type=MediaType.DOCUMENT,
        content=b"fake_pdf_data",
        metadata={"filename": "PRD_v1.pdf"}
    ))

    result3 = agent.process(input3)
    print(f"\n  综合结果:")
    print(f"    摘要: {result3['summary']}")

    # 场景4: 多模态融合
    print("\n" + "=" * 60)
    print("场景4: 多模态融合")
    print("=" * 60)

    input4 = MultimodalInput(context="综合分析所有材料")
    input4.add_text("请综合分析以下材料并给出建议")
    input4.add_image(b"chart_image_data")
    input4.add_audio(b"meeting_recording")
    input4.contents.append(MediaContent(
        media_type=MediaType.DOCUMENT,
        content=b"report_pdf"
    ))

    result4 = agent.process(input4)
    print(f"\n  综合结果:")
    print(f"    摘要: {result4['summary']}")
    print(f"    处理模态: {result4['modalities_processed']}")

    # 统计信息
    print(f"\n{'='*60}")
    print("处理统计")
    print(f"{'='*60}")
    stats = agent.get_stats()
    print(f"  视觉处理: {stats['vision']['processed_count']} 次")
    print(f"  音频处理: {stats['audio']['processed_count']} 次")
    print(f"  文档处理: {stats['document']['processed_count']} 次")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("多模态 Agent")
    print("=" * 60)

    demo_multimodal_agent()

    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了多模态 Agent 的设计与实现：

  核心知识点：
  1. 多模态内容类型：文本/图片/音频/视频/文档
  2. 视觉处理：图片描述、OCR、图表分析
  3. 音频处理：语音转文字、说话人识别、情感分析
  4. 文档处理：PDF 解析、表格提取、关键信息提取
  5. 多模态融合：整合不同模态的处理结果

  实际应用场景：
  - 智能客服：语音+文字+图片理解
  - 文档分析：PDF/表格/图表综合处理
  - 内容审核：多模态内容理解
  - 会议助手：语音转写+摘要生成
    """)
