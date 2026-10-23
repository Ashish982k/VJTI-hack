import argparse
import json
from pathlib import Path

from .config import Settings
from .pipeline import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(description="Smart Knowledge Base for Agents")
    parser.add_argument("--mode", choices=["baseline", "improved"])
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("source", type=Path)
    index.add_argument("--collection", required=True)
    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--collection", required=True)
    sub.add_parser("serve")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.mode:
        from dataclasses import replace
        settings = replace(settings, mode=args.mode)
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(settings), host="0.0.0.0", port=8000)
        return
    kb = KnowledgeBase(settings)
    if args.command == "index":
        result = kb.index_documents([(p.name, p.read_bytes()) for p in sorted(args.source.glob("*.pdf"))], args.collection) if args.source.is_dir() else kb.index_zip(args.source.read_bytes(), args.collection)
    else:
        result = kb.query(args.collection, args.question)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    kb.telemetry.flush()


if __name__ == "__main__":
    main()
