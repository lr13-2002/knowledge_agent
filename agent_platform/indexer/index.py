"""代码索引器核心 — 遍历仓库文件，分发到各语言解析器。

CodeIndexer 是统一入口：
1. 遍历仓库目录（跳过 vendor/node_modules 等）
2. 按文件扩展名分发到 Go/Java/Fallback 解析器
3. 收集所有 symbols 和 chunks
4. 按 (repo_root, commit) 缓存结果，避免重复索引

CodeIndex 是索引结果的数据结构，包含 symbols（符号列表）和 chunks（文本片段列表）。
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .fallback import index_fallback_file
from .go_parser import index_go_file
from .java_parser import index_java_file

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git", ".idea", ".statsd", "kms-log", "log", "data",
    "vendor", "node_modules", "__pycache__", "target", "build", "dist",
}


@dataclass
class CodeIndex:
    repo_name: str
    commit: str = ""
    symbols: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "commit": self.commit,
            "symbols": self.symbols,
            "chunks": self.chunks,
        }


class CodeIndexer:
    """Unified code indexer using tree-sitter (Go) and regex (Java) parsers.

    All parsing happens in-process — no external binaries needed.
    Results are cached per (repo_root, commit).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], CodeIndex] = {}

    def index_repo(self, repo_root: str, repo_name: str | None = None) -> CodeIndex:
        repo_root = os.path.abspath(repo_root)
        if repo_name is None:
            repo_name = os.path.basename(repo_root)
        commit = _git_head(repo_root)
        key = (repo_root, commit)
        if key in self._cache:
            return self._cache[key]

        result = CodeIndex(repo_name=repo_name, commit=commit)

        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel_file = os.path.relpath(abs_path, repo_root).replace("\\", "/")
                ext = os.path.splitext(fname)[1].lower()

                if ext == ".go":
                    syms, chunks = index_go_file(repo_root, rel_file, repo_name)
                    result.symbols.extend(syms)
                    result.chunks.extend(chunks)
                elif ext == ".java":
                    syms, chunks = index_java_file(repo_root, rel_file, repo_name)
                    result.symbols.extend(syms)
                    result.chunks.extend(chunks)
                else:
                    result.chunks.extend(index_fallback_file(repo_root, rel_file, repo_name))

        logger.info(
            "indexed %s: %d symbols, %d chunks (commit=%s)",
            repo_name, len(result.symbols), len(result.chunks),
            commit[:8] if commit else "none",
        )
        self._cache[key] = result
        return result


def _git_head(repo_root: str) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
