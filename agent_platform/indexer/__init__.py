"""代码索引器 — 从仓库源码中提取符号、调用关系和文本片段。

支持三种语言的解析：
- Go: 使用 tree-sitter 做 AST 级解析（函数、方法、结构体、接口、调用关系）
- Java: 使用正则表达式解析（类、方法、调用关系）
- Python/TS/JS: 整文件作为一个 chunk（fallback）

产出的 symbols 和 chunks 会被 RAGLoader 写入图和向量库，
供 Worker 在分析 trace 时检索关联代码。
"""

from .index import CodeIndexer, CodeIndex

__all__ = ["CodeIndexer", "CodeIndex"]
