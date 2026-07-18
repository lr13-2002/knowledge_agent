"""Parser 注册表：从 parsers.yaml 加载规则并按 appname/lang 匹配（AGENTS.md §9.6）。

匹配优先级（先到先得）：
1. match.appname_contains — appname 子串匹配（最具体）
2. match.lang — 语言兜底（java/go）
3. match: {} — generic_fallback,接受所有（L3 终极兜底）

yaml 没配的服务也永远能命中 generic_fallback,保证"未配置也能接入,只是精度降级"。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_DEFAULT_YAML = Path(__file__).parent / "parsers.yaml"


class LogParserRegistry:
    """加载并管理 trace 日志 parser 规则。"""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        # 规则顺序即匹配优先级,generic_fallback（match: {}）应放最后
        self._rules = rules

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "LogParserRegistry":
        """从 yaml 加载。沿用 llm/config.py 的 lazy import 模式,不强加核心依赖。"""
        import yaml  # lazy import,pyyaml 是可选依赖

        yaml_path = Path(path) if path else _DEFAULT_YAML
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        rules = data.get("parsers", [])
        if not rules:
            raise ValueError(f"parsers.yaml 未定义任何 parser 规则: {yaml_path}")
        return cls(rules)

    def pick(self, appname: str, lang: str) -> dict[str, Any]:
        """按 appname/lang 选 parser。永远有返回（最差命中 generic_fallback）。

        appname: span 的 appname[0],如 "hnb-v.cedar.os.biz.didi.com"
        lang: 该 repo 的语言（"go"/"java"）,真实实现从代码索引的 repo→lang 反查得到
        """
        appname_lc = (appname or "").lower()
        fallback: dict[str, Any] | None = None
        for rule in self._rules:
            match = rule.get("match") or {}
            if "appname_contains" in match:
                if match["appname_contains"].lower() in appname_lc:
                    return rule
            elif "lang" in match:
                if match["lang"] == lang:
                    return rule
            elif not match:
                # 空 match = generic_fallback,记下来但继续找更具体的
                fallback = fallback or rule
        if fallback is not None:
            return fallback
        # 没有 generic_fallback 配置时,返回最后一条规则兜底
        return self._rules[-1]
