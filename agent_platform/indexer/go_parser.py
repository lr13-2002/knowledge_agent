"""Go 代码解析器 — 基于 tree-sitter 的 AST 级解析。

提取内容：
- 函数声明（function_declaration）→ 符号 + 调用关系
- 方法声明（method_declaration）→ 符号 + receiver 类型 + 调用关系
- 类型声明（struct/interface）→ 符号

每个符号生成一个 chunk，包含 qualified_name、签名、调用列表等信息，
写入向量库后可通过语义检索找到。

tree-sitter 未安装时自动降级为整文件 chunk（不解析 AST）。
"""
from __future__ import annotations

import os
from typing import Any

try:
    import tree_sitter_go as _tsgo
    from tree_sitter import Language, Parser
    _GO_LANG = Language(_tsgo.language())
    _HAS_TREESITTER = True
except ImportError:
    _HAS_TREESITTER = False


def index_go_file(repo_root: str, rel_file: str, repo_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    abs_path = os.path.join(repo_root, rel_file)
    if not os.path.isfile(abs_path):
        return [], []
    if rel_file.endswith("_test.go"):
        return [], []
    if not _HAS_TREESITTER:
        return [], [_fallback_chunk(abs_path, rel_file, repo_name)]
    with open(abs_path, "rb") as f:
        source = f.read()
    parser = Parser(_GO_LANG)
    tree = parser.parse(source)
    module = _module_for_file(rel_file)
    symbols: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for node in tree.root_node.children:
        if node.type == "function_declaration":
            sym = _extract_function(node, source, rel_file, module, repo_name)
            symbols.append(sym)
            chunks.append(_chunk_from_symbol(sym, repo_name))
        elif node.type == "method_declaration":
            sym = _extract_method(node, source, rel_file, module, repo_name)
            symbols.append(sym)
            chunks.append(_chunk_from_symbol(sym, repo_name))
        elif node.type == "type_declaration":
            for spec in node.children:
                if spec.type == "type_spec":
                    sym = _extract_type(spec, source, rel_file, module, repo_name)
                    symbols.append(sym)
                    chunks.append(_chunk_from_symbol(sym, repo_name))

    return symbols, chunks


def _extract_function(node: Any, source: bytes, rel_file: str, module: str, repo_name: str) -> dict[str, Any]:
    name_node = node.child_by_field_name("name")
    name = _text(source, name_node) if name_node else "?"
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    sig = _text(source, node).split("{")[0].strip()
    body = node.child_by_field_name("body")
    calls = _extract_calls(body, source, rel_file) if body else []
    sym_id = f"go:{rel_file}:{name}:{start}"
    return {
        "id": sym_id,
        "language": "go",
        "symbol_type": "function",
        "symbol_name": name,
        "qualified_name": _qualified_name(repo_name, module, name),
        "module": module,
        "file": rel_file,
        "start_line": start,
        "end_line": end,
        "signature": sig,
        "annotations": [],
        "calls": calls,
        "chunk_id": f"code:{sym_id}",
    }


def _extract_method(node: Any, source: bytes, rel_file: str, module: str, repo_name: str) -> dict[str, Any]:
    name_node = node.child_by_field_name("name")
    name = _text(source, name_node) if name_node else "?"
    receiver = _extract_receiver(node, source)
    if receiver:
        name = f"{receiver}.{name}"
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    sig = _text(source, node).split("{")[0].strip()
    body = node.child_by_field_name("body")
    calls = _extract_calls(body, source, rel_file) if body else []
    sym_id = f"go:{rel_file}:{name}:{start}"
    return {
        "id": sym_id,
        "language": "go",
        "symbol_type": "function",
        "symbol_name": name,
        "qualified_name": _qualified_name(repo_name, module, name),
        "module": module,
        "file": rel_file,
        "start_line": start,
        "end_line": end,
        "signature": sig,
        "annotations": [],
        "calls": calls,
        "chunk_id": f"code:{sym_id}",
    }


def _extract_type(spec: Any, source: bytes, rel_file: str, module: str, repo_name: str) -> dict[str, Any]:
    name_node = spec.child_by_field_name("name")
    name = _text(source, name_node) if name_node else "?"
    type_node = spec.child_by_field_name("type")
    kind = "type"
    if type_node:
        if type_node.type == "struct_type":
            kind = "struct"
        elif type_node.type == "interface_type":
            kind = "interface"
    start = spec.start_point[0] + 1
    end = spec.end_point[0] + 1
    sig = _text(source, spec)
    sym_id = f"go:{rel_file}:{name}:{start}"
    return {
        "id": sym_id,
        "language": "go",
        "symbol_type": kind,
        "symbol_name": name,
        "qualified_name": _qualified_name(repo_name, module, name),
        "module": module,
        "file": rel_file,
        "start_line": start,
        "end_line": end,
        "signature": sig,
        "annotations": [],
        "calls": [],
        "chunk_id": f"code:{sym_id}",
    }


def _extract_receiver(node: Any, source: bytes) -> str:
    receiver = node.child_by_field_name("receiver")
    if not receiver:
        return ""
    for param in _iter_type(receiver, "parameter_declaration"):
        type_node = param.child_by_field_name("type")
        if type_node:
            t = _text(source, type_node).lstrip("*")
            return t
    return ""


def _iter_type(node: Any, type_name: str) -> list[Any]:
    results = []
    if node.type == type_name:
        results.append(node)
    for child in node.children:
        results.extend(_iter_type(child, type_name))
    return results


def _extract_calls(node: Any, source: bytes, rel_file: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    _walk_calls(node, source, rel_file, calls, seen)
    calls.sort(key=lambda c: (c["line"], c["callee"]))
    return calls


def _walk_calls(node: Any, source: bytes, rel_file: str, calls: list[dict[str, Any]], seen: set[str]) -> None:
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn:
            callee = _text(source, fn)
            line = node.start_point[0] + 1
            args_node = node.child_by_field_name("arguments")
            argc = sum(1 for c in (args_node.children if args_node else []) if c.type not in ("(", ")", ","))
            key = f"{callee}:{line}:{argc}"
            if callee and key not in seen:
                seen.add(key)
                calls.append({"callee": callee, "file": rel_file, "line": line, "args": argc})
    for child in node.children:
        _walk_calls(child, source, rel_file, calls, seen)


def _chunk_from_symbol(sym: dict[str, Any], repo_name: str) -> dict[str, Any]:
    call_names = sorted({c["callee"] for c in sym.get("calls", [])})
    return {
        "id": sym.get("chunk_id", f"code:{sym['id']}"),
        "kind": "code_symbol",
        "repo": repo_name,
        "language": "go",
        "symbol_id": sym["id"],
        "symbol_name": sym["symbol_name"],
        "qualified_name": sym["qualified_name"],
        "module": sym["module"],
        "file": sym["file"],
        "start_line": sym["start_line"],
        "end_line": sym["end_line"],
        "text": (
            f"language=go module={sym['module']} file={sym['file']} "
            f"lines={sym['start_line']}-{sym['end_line']} "
            f"symbol_type={sym['symbol_type']} qualified_name={sym['qualified_name']} "
            f"signature={sym['signature']} calls={', '.join(call_names)}"
        ),
        "calls": call_names,
    }


def _text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _module_for_file(path: str) -> str:
    d = os.path.dirname(path).replace("\\", "/")
    return d if d and d != "." else ""


def _qualified_name(repo_name: str, module: str, name: str) -> str:
    if not module:
        return f"{repo_name}.{name}"
    return f"{repo_name}.{module.replace('/', '.')}.{name}"


def _fallback_chunk(abs_path: str, rel_file: str, repo_name: str) -> dict[str, Any]:
    with open(abs_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")
    return {
        "id": f"file:{rel_file}",
        "kind": "file",
        "repo": repo_name,
        "language": "go",
        "file": rel_file,
        "start_line": 1,
        "end_line": len(lines),
        "text": f"language=go file={rel_file} lines=1-{len(lines)}\n{text}",
    }
