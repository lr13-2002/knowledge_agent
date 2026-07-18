"""Fallback 解析器 — 将不支持 AST 解析的文件整体作为一个 chunk。

支持: .py, .ts, .tsx, .js, .jsx
不做符号提取，只把整个文件文本写入向量库，靠语义检索命中。
"""
from __future__ import annotations

import os
from typing import Any

_LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
}


def detect_language(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _LANGUAGE_MAP.get(ext, "")


def index_fallback_file(repo_root: str, rel_file: str, repo_name: str) -> list[dict[str, Any]]:
    abs_path = os.path.join(repo_root, rel_file)
    if not os.path.isfile(abs_path):
        return []
    language = detect_language(rel_file)
    if not language:
        return []
    with open(abs_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")
    return [{
        "id": f"file:{rel_file}",
        "kind": "file",
        "repo": repo_name,
        "language": language,
        "file": rel_file,
        "start_line": 1,
        "end_line": len(lines),
        "text": f"language={language} file={rel_file} lines=1-{len(lines)}\n{text}",
    }]
