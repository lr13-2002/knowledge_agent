"""Java 代码解析器 — 基于正则表达式的轻量级解析。

提取内容：
- 类/接口/枚举声明 → 符号
- 方法声明 → 符号 + 调用关系
- 方法体内的函数调用 → calls 列表

使用正则而非 AST 的原因：Java 的 tree-sitter 绑定在某些环境下不稳定，
正则对于提取类名、方法签名、调用关系已经够用。
"""
from __future__ import annotations

import os
import re
from typing import Any

_CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_]\w*)")
_METHOD_RE = re.compile(
    r"(?:public|protected|private|static|final|synchronized|abstract|\s)+"
    r"[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:throws [^{]+)?\{"
)
_CALL_RE = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "new", "throw"}
_PKG_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z0-9_.]+)\s*;")


def index_java_file(repo_root: str, rel_file: str, repo_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    abs_path = os.path.join(repo_root, rel_file)
    if not os.path.isfile(abs_path):
        return [], []
    with open(abs_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")
    module = _module_for_file(rel_file)
    pkg = _first_package(text)
    symbols: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    current_class = ""
    annotations: list[str] = []

    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("@"):
            annotations.append(trimmed.split()[0])
            continue

        class_match = _CLASS_RE.search(trimmed)
        if class_match:
            current_class = class_match.group(2)
            start = i + 1
            end = _find_brace_end(lines, i)
            sym_id = f"java:{rel_file}:{current_class}:{start}"
            sym = {
                "id": sym_id,
                "language": "java",
                "symbol_type": class_match.group(1),
                "symbol_name": current_class,
                "qualified_name": f"{pkg}.{current_class}" if pkg else current_class,
                "module": module,
                "file": rel_file,
                "start_line": start,
                "end_line": end,
                "signature": trimmed,
                "annotations": list(annotations),
                "calls": [],
            }
            annotations = []
            symbols.append(sym)
            chunks.append(_chunk_from_symbol(sym, repo_name))
            continue

        method_match = _METHOD_RE.search(trimmed)
        if method_match and current_class and " class " not in trimmed:
            method_name = method_match.group(1)
            start = i + 1
            end = _find_brace_end(lines, i)
            name = f"{current_class}.{method_name}"
            sym_id = f"java:{rel_file}:{name}:{start}"
            calls = _extract_calls(lines[i:end], rel_file, start)
            sym = {
                "id": sym_id,
                "language": "java",
                "symbol_type": "method",
                "symbol_name": name,
                "qualified_name": f"{pkg}.{name}" if pkg else name,
                "module": module,
                "file": rel_file,
                "start_line": start,
                "end_line": end,
                "signature": trimmed,
                "annotations": list(annotations),
                "calls": calls,
            }
            annotations = []
            symbols.append(sym)
            chunks.append(_chunk_from_symbol(sym, repo_name))

        if trimmed and not trimmed.startswith("@"):
            annotations = []

    return symbols, chunks


def _find_brace_end(lines: list[str], start: int) -> int:
    depth = 0
    seen = False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
                if seen and depth <= 0:
                    return i + 1
    return start + 1


def _extract_calls(lines: list[str], rel_file: str, start_line: int) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        for m in _CALL_RE.finditer(line):
            callee = m.group(1)
            if callee in _KEYWORDS or callee.startswith("public") or callee.startswith("private"):
                continue
            key = f"{callee}:{start_line + i}"
            if key in seen:
                continue
            seen.add(key)
            calls.append({"callee": callee, "file": rel_file, "line": start_line + i})
    return calls


def _chunk_from_symbol(sym: dict[str, Any], repo_name: str) -> dict[str, Any]:
    call_names = sorted({c["callee"] for c in sym.get("calls", [])})
    return {
        "id": f"code:{sym['id']}",
        "kind": "code_symbol",
        "repo": repo_name,
        "language": "java",
        "symbol_id": sym["id"],
        "symbol_name": sym["symbol_name"],
        "qualified_name": sym["qualified_name"],
        "module": sym["module"],
        "file": sym["file"],
        "start_line": sym["start_line"],
        "end_line": sym["end_line"],
        "text": (
            f"language=java module={sym['module']} file={sym['file']} "
            f"lines={sym['start_line']}-{sym['end_line']} "
            f"symbol_type={sym['symbol_type']} qualified_name={sym['qualified_name']} "
            f"signature={sym['signature']} calls={', '.join(call_names)}"
        ),
        "calls": call_names,
    }


def _first_package(text: str) -> str:
    m = _PKG_RE.search(text)
    return m.group(1) if m else ""


def _module_for_file(path: str) -> str:
    d = os.path.dirname(path).replace("\\", "/")
    return d if d and d != "." else ""
