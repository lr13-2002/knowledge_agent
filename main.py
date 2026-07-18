"""知识库智能体平台 HTTP 服务启动入口。

用法:
    python main.py                                        # 默认 127.0.0.1:8000
    python main.py --port 8000 --host 0.0.0.0             # 自定义地址
    python main.py --repo my-service=/path/to/repo        # 映射本地代码仓库
    python main.py --data-dir /data/knowledge             # 自定义数据存储目录
"""
from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Business Understanding Agent Platform")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument(
        "--repo",
        action="append",
        metavar="NAME=PATH",
        help="Map a repo name to a local path, e.g. --repo my-service=/src/my-service",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "data"),
        help="Directory for persistent storage (default: data/)",
    )
    return parser.parse_args(argv)


def build_repo_roots(raw: list[str] | None) -> dict[str, str]:
    roots: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            print(f"warning: ignoring --repo {item!r} (expected NAME=PATH)", file=sys.stderr)
            continue
        name, path = item.split("=", 1)
        roots[name.strip()] = os.path.expanduser(path.strip())
    return roots


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_roots = build_repo_roots(args.repo)

    from agent_platform.api import create_app

    app = create_app(repo_roots=repo_roots, data_dir=args.data_dir)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
