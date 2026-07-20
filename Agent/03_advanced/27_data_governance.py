"""
==============================================================================
第二十七课（进阶）：Data Governance & Privacy（数据治理与隐私）
==============================================================================

【为什么需要单独一课？】
现有课程没有系统讲解 Agent 系统中的数据治理。
生产级 Agent 必须处理 PII、数据分类、合规性等问题。

【学习目标】
- 理解 Agent 系统面临的数据治理挑战
- 掌握 PII（个人身份信息）检测与脱敏
- 掌握数据分类分级
- 学会数据访问控制
- 理解合规要求（GDPR/个人信息保护法）

【核心概念】
- PII Detection（个人身份信息检测）
- Data Classification（数据分类）
- Data Masking（数据脱敏）
- Access Control（访问控制）
- Compliance（合规）
- Data Retention（数据保留策略）

==============================================================================
"""

import hashlib
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 第一部分：数据治理概览
# ============================================================================

def explain_data_governance_overview():
    """
    Agent 系统的数据治理挑战

    Agent 系统处理大量敏感数据：
    - 用户对话内容（可能包含个人信息）
    - 工具调用结果（可能包含机密数据）
    - 记忆存储（持久化的敏感信息）
    - 日志记录（可能泄露隐私）
    """

    print("=" * 60)
    print("Data Governance & Privacy (数据治理与隐私) 概览")
    print("=" * 60)

    print("""
  -- Agent 系统的数据治理挑战 --

  1. 输入数据不可控:
     用户可能在对话中提供身份证号、银行卡号等敏感信息

  2. LLM 处理不透明:
     数据发送给 LLM 后，无法保证不被记录或泄露

  3. 记忆持久化风险:
     敏感信息被存入长期记忆，增加泄露面

  4. 日志泄露:
     日志中可能包含完整的对话内容和工具结果

  5. 合规要求:
     GDPR/个人信息保护法要求数据最小化、可删除


  -- 数据治理的核心原则 --

  1. 数据最小化: 只收集和处理必要的数据
  2. 目的限制: 数据只用于声明的目的
  3. 存储限制: 数据不无限期保留
  4. 完整性与机密性: 保护数据不被未授权访问
  5. 可问责性: 能证明合规措施已实施
    """)


# ============================================================================
# 第二部分：数据分类分级
# ============================================================================

class DataClassification(Enum):
    """数据分类"""
    PUBLIC = "public"               # 公开数据
    INTERNAL = "internal"           # 内部数据
    CONFIDENTIAL = "confidential"   # 机密数据
    RESTRICTED = "restricted"       # 受限数据（PII/敏感）
    CRITICAL = "critical"           # 关键数据（密钥/凭证）


@dataclass
class DataClassificationRule:
    """数据分类规则"""
    name: str
    classification: DataClassification
    patterns: List[str] = field(default_factory=list)
    description: str = ""


class DataClassifier:
    """
    数据分类器

    【原理】
    根据数据内容和模式，自动分类数据级别。
    不同级别的数据有不同的处理规则。
    """

    DEFAULT_RULES = [
        DataClassificationRule(
            name="PII-身份证",
            classification=DataClassification.RESTRICTED,
            patterns=[r"\d{17}[\dXx]", r"\d{15}"],
            description="身份证号码",
        ),
        DataClassificationRule(
            name="PII-手机号",
            classification=DataClassification.RESTRICTED,
            patterns=[r"1[3-9]\d{9}"],
            description="手机号码",
        ),
        DataClassificationRule(
            name="PII-银行卡",
            classification=DataClassification.RESTRICTED,
            patterns=[r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"],
            description="银行卡号",
        ),
        DataClassificationRule(
            name="PII-邮箱",
            classification=DataClassification.CONFIDENTIAL,
            patterns=[r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
            description="电子邮箱",
        ),
        DataClassificationRule(
            name="凭证-API Key",
            classification=DataClassification.CRITICAL,
            patterns=[r"api[_-]?key\s*[:=]\s*\S+", r"sk-[a-zA-Z0-9]{20,}"],
            description="API 密钥",
        ),
        DataClassificationRule(
            name="凭证-密码",
            classification=DataClassification.CRITICAL,
            patterns=[r"password\s*[:=]\s*\S+", r"passwd\s*[:=]\s*\S+"],
            description="密码",
        ),
    ]

    def __init__(self, custom_rules: List[DataClassificationRule] = None):
        self.rules = custom_rules or self.DEFAULT_RULES.copy()
        self._compiled_rules = [
            (rule, [re.compile(p, re.IGNORECASE) for p in rule.patterns])
            for rule in self.rules
        ]

    def classify(self, text: str) -> Dict[str, Any]:
        """
        分类数据

        Returns:
            {
                "classification": DataClassification,
                "matched_rules": List[str],
                "findings": List[Dict],
                "risk_score": float
            }
        """
        matched_rules = []
        findings = []
        max_classification = DataClassification.PUBLIC

        for rule, patterns in self._compiled_rules:
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    matched_rules.append(rule.name)
                    findings.append({
                        "rule": rule.name,
                        "classification": rule.classification.value,
                        "matches_count": len(matches),
                        "description": rule.description,
                    })
                    # 取最高分类级别
                    if self._level_order(rule.classification) > self._level_order(max_classification):
                        max_classification = rule.classification
                    break

        # 计算风险分数
        risk_score = min(1.0, len(findings) * 0.3)

        return {
            "classification": max_classification,
            "matched_rules": matched_rules,
            "findings": findings,
            "risk_score": risk_score,
        }

    def _level_order(self, classification: DataClassification) -> int:
        order = {
            DataClassification.PUBLIC: 0,
            DataClassification.INTERNAL: 1,
            DataClassification.CONFIDENTIAL: 2,
            DataClassification.RESTRICTED: 3,
            DataClassification.CRITICAL: 4,
        }
        return order.get(classification, 0)


# ============================================================================
# 第三部分：PII 检测与脱敏
# ============================================================================

class PIIDetector:
    """
    PII 检测器

    【功能】
    检测文本中的个人身份信息（PII）：
    - 身份证号
    - 手机号
    - 银行卡号
    - 邮箱地址
    - 姓名（基于模式匹配）
    - IP 地址
    """

    PII_PATTERNS = {
        "身份证": r"\b(\d{17}[\dXx])\b",
        "手机号": r"\b(1[3-9]\d{9})\b",
        "银行卡": r"\b(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",
        "邮箱": r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
        "IP地址": r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",
    }

    def __init__(self):
        self._compiled = {
            name: re.compile(pattern)
            for name, pattern in self.PII_PATTERNS.items()
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        检测 PII

        Returns:
            检测到的 PII 列表
        """
        findings = []
        for pii_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": pii_type,
                    "value": match.group(1),
                    "start": match.start(),
                    "end": match.end(),
                })
        return findings

    def mask(self, text: str) -> Tuple[str, List[Dict]]:
        """
        脱敏处理

        Returns:
            (脱敏后的文本, 脱敏记录)
        """
        findings = self.detect(text)
        masked_text = text
        mask_records = []

        # 按位置倒序替换，避免偏移
        for finding in sorted(findings, key=lambda x: x["start"], reverse=True):
            value = finding["value"]
            pii_type = finding["type"]

            # 根据类型选择脱敏方式
            if pii_type == "身份证":
                masked = value[:3] + "*" * (len(value) - 4) + value[-1]
            elif pii_type == "手机号":
                masked = value[:3] + "****" + value[-4:]
            elif pii_type == "银行卡":
                masked = "****-****-****-" + value[-4:]
            elif pii_type == "邮箱":
                name_part = value.split("@")[0]
                domain = value.split("@")[1]
                masked = name_part[:2] + "***@" + domain
            elif pii_type == "IP地址":
                parts = value.split(".")
                masked = f"{parts[0]}.{parts[1]}.*.*"
            else:
                masked = "*" * len(value)

            masked_text = (
                masked_text[:finding["start"]]
                + masked
                + masked_text[finding["end"]:]
            )
            mask_records.append({
                "type": pii_type,
                "original_hash": hashlib.md5(value.encode()).hexdigest()[:8],
                "masked": masked,
            })

        return masked_text, mask_records


# ============================================================================
# 第四部分：数据访问控制
# ============================================================================

class AccessLevel(Enum):
    """访问级别"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    NONE = "none"


@dataclass
class AccessPolicy:
    """访问策略"""
    role: str
    resource_pattern: str
    allowed_levels: List[AccessLevel]
    conditions: Dict[str, Any] = field(default_factory=dict)


class AccessControlManager:
    """
    访问控制管理器

    【原理】
    基于角色的访问控制（RBAC）：
    - 定义角色和权限
    - 检查请求是否符合策略
    - 支持条件判断

    【Agent 场景】
    - 不同用户角色访问不同的 Agent 功能
    - 限制敏感工具的使用权限
    - 控制记忆数据的访问范围
    """

    def __init__(self):
        self._policies: List[AccessPolicy] = []
        self._role_assignments: Dict[str, List[str]] = {}  # user_id -> roles
        self._audit_log: List[Dict] = []

    def add_policy(self, policy: AccessPolicy) -> None:
        """添加策略"""
        self._policies.append(policy)

    def assign_role(self, user_id: str, roles: List[str]) -> None:
        """分配角色"""
        self._role_assignments[user_id] = roles

    def check_access(
        self, user_id: str, resource: str, level: AccessLevel
    ) -> Dict[str, Any]:
        """
        检查访问权限

        Returns:
            {
                "allowed": bool,
                "reason": str,
                "matched_policy": Optional[str]
            }
        """
        user_roles = self._role_assignments.get(user_id, [])

        for policy in self._policies:
            if policy.role in user_roles:
                if re.match(policy.resource_pattern, resource):
                    if level in policy.allowed_levels:
                        self._log_access(user_id, resource, level, True)
                        return {
                            "allowed": True,
                            "reason": f"角色 {policy.role} 有 {level.value} 权限",
                            "matched_policy": policy.name if hasattr(policy, 'name') else str(policy),
                        }

        self._log_access(user_id, resource, level, False)
        return {
            "allowed": False,
            "reason": f"用户 {user_id} 无权访问 {resource} ({level.value})",
            "matched_policy": None,
        }

    def _log_access(
        self, user_id: str, resource: str, level: AccessLevel, allowed: bool
    ) -> None:
        """记录审计日志"""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "resource": resource,
            "level": level.value,
            "allowed": allowed,
        })

    def get_audit_log(self, user_id: Optional[str] = None) -> List[Dict]:
        """获取审计日志"""
        if user_id:
            return [e for e in self._audit_log if e["user_id"] == user_id]
        return list(self._audit_log)


# ============================================================================
# 第五部分：数据保留策略
# ============================================================================

class RetentionPolicy:
    """
    数据保留策略

    【原理】
    不同类型的数据有不同的保留期限：
    - 对话记录: 保留 N 天后删除
    - 记忆数据: 用户可主动删除
    - 日志数据: 保留 M 天后归档
    - 审计日志: 永久保留（合规要求）
    """

    def __init__(self):
        self._rules: Dict[str, timedelta] = {
            "conversation": timedelta(days=30),
            "memory": timedelta(days=365),
            "log": timedelta(days=90),
            "audit": timedelta(days=2555),  # 7年
        }
        self._records: List[Dict] = []

    def set_retention(self, data_type: str, duration: timedelta) -> None:
        """设置保留期限"""
        self._rules[data_type] = duration

    def add_record(self, data_type: str, record_id: str, 
                   created_at: datetime = None) -> None:
        """添加数据记录"""
        self._records.append({
            "data_type": data_type,
            "record_id": record_id,
            "created_at": created_at or datetime.now(),
            "expires_at": (created_at or datetime.now()) + self._rules.get(
                data_type, timedelta(days=30)
            ),
            "status": "active",
        })

    def check_expiry(self) -> List[Dict]:
        """检查过期数据"""
        now = datetime.now()
        expired = []
        for record in self._records:
            if record["status"] == "active" and now > record["expires_at"]:
                record["status"] = "expired"
                expired.append(record)
        return expired

    def cleanup(self) -> int:
        """清理过期数据"""
        expired = self.check_expiry()
        count = len(expired)
        self._records = [r for r in self._records if r["status"] == "active"]
        return count

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        by_type = {}
        for record in self._records:
            dt = record["data_type"]
            if dt not in by_type:
                by_type[dt] = {"active": 0, "expires_soon": 0}
            by_type[dt]["active"] += 1
            if record["expires_at"] - datetime.now() < timedelta(days=7):
                by_type[dt]["expires_soon"] += 1
        return by_type


# ============================================================================
# 第六部分：数据治理管道
# ============================================================================

class DataGovernancePipeline:
    """
    数据治理管道

    【功能】
    将分类、检测、脱敏、访问控制整合为统一管道：
    1. 输入数据 -> 分类
    2. 分类结果 -> PII 检测
    3. PII 结果 -> 脱敏处理
    4. 脱敏后数据 -> 访问控制检查
    5. 通过 -> 传递给下游
    """

    def __init__(self):
        self.classifier = DataClassifier()
        self.pii_detector = PIIDetector()
        self.access_manager = AccessControlManager()
        self.retention = RetentionPolicy()
        self._processed_count = 0
        self._blocked_count = 0

    def process_input(self, text: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        处理输入数据

        Returns:
            {
                "original": str,
                "processed": str,
                "classification": str,
                "pii_found": bool,
                "pii_masked": bool,
                "access_allowed": bool,
                "audit_records": List
            }
        """
        self._processed_count += 1
        result = {
            "original": text,
            "processed": text,
            "classification": "public",
            "pii_found": False,
            "pii_masked": False,
            "access_allowed": True,
            "audit_records": [],
        }

        # Step 1: 数据分类
        classification = self.classifier.classify(text)
        result["classification"] = classification["classification"].value

        if classification["classification"] == DataClassification.CRITICAL:
            # 关键数据直接拦截
            result["access_allowed"] = False
            result["processed"] = "[已拦截: 包含敏感凭证信息]"
            self._blocked_count += 1
            return result

        # Step 2: PII 检测
        pii_findings = self.pii_detector.detect(text)
        result["pii_found"] = len(pii_findings) > 0

        # Step 3: PII 脱敏
        if pii_findings:
            masked_text, mask_records = self.pii_detector.mask(text)
            result["processed"] = masked_text
            result["pii_masked"] = True
            result["audit_records"].extend(mask_records)

        # Step 4: 访问控制
        access_result = self.access_manager.check_access(
            user_id, "agent.input", AccessLevel.READ
        )
        result["access_allowed"] = access_result["allowed"]

        return result

    def get_stats(self) -> Dict[str, int]:
        return {
            "processed": self._processed_count,
            "blocked": self._blocked_count,
        }


# ============================================================================
# 第七部分：完整演示
# ============================================================================

def demonstrate_classification():
    """演示数据分类"""

    print("\n" + "=" * 60)
    print("演示1: 数据分类")
    print("=" * 60)

    classifier = DataClassifier()

    test_cases = [
        ("你好，今天天气不错", "普通对话"),
        ("我的邮箱是 zhang@example.com", "包含邮箱"),
        ("身份证号: 110101199001011234", "包含身份证"),
        ("API Key: sk-abc123def456ghi789jkl012mno", "包含密钥"),
        ("银行卡号: 6222-0200-1234-5678", "包含银行卡"),
    ]

    for text, desc in test_cases:
        result = classifier.classify(text)
        print(f"\n  [{desc}] {text[:40]}...")
        print(f"    分类: {result['classification'].value}")
        print(f"    风险: {result['risk_score']:.1f}")
        if result["findings"]:
            for f in result["findings"]:
                print(f"    发现: {f['rule']} ({f['matches_count']}处)")


def demonstrate_pii_masking():
    """演示 PII 脱敏"""

    print("\n" + "=" * 60)
    print("演示2: PII 检测与脱敏")
    print("=" * 60)

    detector = PIIDetector()

    test_texts = [
        "用户张三的手机号是13812345678，邮箱是zhangsan@example.com",
        "请转账到银行卡6222020012345678，持卡人李四",
        "服务器IP: 192.168.1.100，管理员邮箱admin@company.com",
    ]

    for text in test_texts:
        print(f"\n  原文: {text}")

        # 检测
        findings = detector.detect(text)
        print(f"    检测到 {len(findings)} 个 PII:")
        for f in findings:
            print(f"      - {f['type']}: {f['value']}")

        # 脱敏
        masked, records = detector.mask(text)
        print(f"    脱敏后: {masked}")


def demonstrate_access_control():
    """演示访问控制"""

    print("\n" + "=" * 60)
    print("演示3: 访问控制")
    print("=" * 60)

    acm = AccessControlManager()

    # 配置策略
    acm.add_policy(AccessPolicy(
        role="admin",
        resource_pattern=".*",
        allowed_levels=[AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN],
    ))
    acm.add_policy(AccessPolicy(
        role="user",
        resource_pattern="agent\\.input",
        allowed_levels=[AccessLevel.READ],
    ))
    acm.add_policy(AccessPolicy(
        role="user",
        resource_pattern="memory\\..*",
        allowed_levels=[AccessLevel.READ],
    ))

    # 分配角色
    acm.assign_role("user_001", ["user"])
    acm.assign_role("admin_001", ["admin"])

    # 测试访问
    tests = [
        ("user_001", "agent.input", AccessLevel.READ),
        ("user_001", "agent.config", AccessLevel.WRITE),
        ("admin_001", "agent.config", AccessLevel.ADMIN),
        ("user_001", "memory.user_001", AccessLevel.READ),
    ]

    for user_id, resource, level in tests:
        result = acm.check_access(user_id, resource, level)
        status = "PASS" if result["allowed"] else "DENY"
        print(f"\n  [{status}] {user_id} -> {resource} ({level.value})")
        print(f"    原因: {result['reason']}")


def demonstrate_governance_pipeline():
    """演示数据治理管道"""

    print("\n" + "=" * 60)
    print("演示4: 数据治理管道")
    print("=" * 60)

    pipeline = DataGovernancePipeline()

    # 配置默认策略
    pipeline.access_manager.add_policy(AccessPolicy(
        role="user",
        resource_pattern=".*",
        allowed_levels=[AccessLevel.READ, AccessLevel.WRITE],
    ))
    pipeline.access_manager.assign_role("user_001", ["user"])

    test_inputs = [
        ("帮我查一下天气", "user_001"),
        ("我的手机号是13812345678，帮我注册", "user_001"),
        ("API Key是sk-abc123def456ghi789jkl012mno", "user_001"),
        ("身份证号110101199001011234，邮箱test@mail.com", "user_001"),
    ]

    for text, user_id in test_inputs:
        print(f"\n  输入: {text[:50]}...")
        result = pipeline.process_input(text, user_id)
        print(f"    分类: {result['classification']}")
        print(f"    PII: {result['pii_found']}")
        print(f"    脱敏: {result['pii_masked']}")
        print(f"    允许: {result['access_allowed']}")
        if result["processed"] != text:
            print(f"    处理后: {result['processed'][:50]}...")

    stats = pipeline.get_stats()
    print(f"\n  -- 管道统计 --")
    print(f"    已处理: {stats['processed']}")
    print(f"    已拦截: {stats['blocked']}")


def main():
    """主函数"""
    print("=" * 60)
    print("第27课: Data Governance & Privacy (数据治理与隐私)")
    print("=" * 60)

    # 1. 概览
    explain_data_governance_overview()

    # 2. 数据分类
    demonstrate_classification()

    # 3. PII 脱敏
    demonstrate_pii_masking()

    # 4. 访问控制
    demonstrate_access_control()

    # 5. 治理管道
    demonstrate_governance_pipeline()

    # 总结
    print("\n" + "=" * 60)
    print("课程总结")
    print("=" * 60)
    print("""
  本课介绍了 Agent 系统的数据治理与隐私保护：

  核心能力：
  1. 数据分类: 自动识别数据级别（公开/内部/机密/受限/关键）
  2. PII 检测: 识别身份证、手机号、银行卡、邮箱等
  3. 数据脱敏: 对敏感信息进行掩码处理
  4. 访问控制: 基于角色的权限管理（RBAC）
  5. 数据保留: 自动过期清理

  合规要点：
  - 数据最小化: 只收集必要数据
  - 目的限制: 数据只用于声明目的
  - 存储限制: 设置保留期限
  - 可删除性: 支持用户数据删除请求
  - 审计追踪: 记录所有数据访问

  最佳实践：
  - 输入数据先分类再处理
  - 敏感信息在日志中脱敏
  - 关键数据（密钥等）直接拦截
  - 定期审计数据访问记录
  - 设置合理的数据保留策略
""")


if __name__ == "__main__":
    main()
