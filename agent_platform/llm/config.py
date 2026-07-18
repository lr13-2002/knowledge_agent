"""LLM 配置。

支持三种配置方式（优先级从高到低）：
1. 代码传参: AnthropicLLMClient(config=LLMConfig(...))
2. 配置文件: llm_config.yaml
3. 环境变量: ANTHROPIC_AUTH_TOKEN 等
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """LLM 调用配置。"""
    # API 连接
    api_key: str = ""  # API 密钥
    base_url: str = ""  # 代理地址（如公司 LiteLLM 代理）
    model: str = "auto-max"  # 模型名

    # 调用参数
    max_tokens: int = 4096  # 最大输出 token
    temperature: float = 0.0  # 温度（0=确定性输出，知识生成场景建议 0）
    max_retries: int = 2  # 校验失败重试次数
    timeout: float = 60.0  # 单次调用超时（秒）

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量构造配置。"""
        return cls(
            api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY") or "",
            base_url=os.environ.get("ANTHROPIC_BASE_URL") or "",
            model=os.environ.get("LLM_MODEL") or "auto-max",
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
            timeout=float(os.environ.get("LLM_TIMEOUT", "60.0")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMConfig":
        """从 YAML 配置文件加载，空值自动 fallback 到环境变量。"""
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        llm = data.get("llm", data)
        env = cls.from_env()
        return cls(
            api_key=str(llm.get("api_key", "")) or env.api_key,
            base_url=str(llm.get("base_url", "")) or env.base_url,
            model=str(llm.get("model", "")) or env.model,
            max_tokens=int(llm.get("max_tokens", 0)) or env.max_tokens,
            temperature=float(llm.get("temperature", 0.0)),
            max_retries=int(llm.get("max_retries", 0)) or env.max_retries,
            timeout=float(llm.get("timeout", 0)) or env.timeout,
        )

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "LLMConfig":
        """自动加载配置：有配置文件用文件，没有用环境变量。

        查找顺序：
            1. 指定的 config_path
            2. 当前目录的 llm_config.yaml
            3. 环境变量
        """
        if config_path and Path(config_path).exists():
            return cls.from_yaml(config_path)
        default_path = Path("llm_config.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls.from_env()
