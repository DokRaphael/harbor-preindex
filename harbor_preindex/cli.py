"""CLI interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from harbor_preindex import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the root CLI parser."""

    parser = argparse.ArgumentParser(prog="harbor-preindex")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    build_index_parser = subparsers.add_parser(
        "build-index", help="Build or update the project index"
    )
    build_index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing",
    )

    subparsers.add_parser("rescan", help="Recreate the collection and rebuild the index")

    query_file_parser = subparsers.add_parser(
        "query-file", help="Find the best project folder for a file"
    )
    query_file_parser.add_argument("file_path", help="Path to the input file")
    query_file_parser.add_argument(
        "--top-k", type=int, default=None, help="Override top-k retrieval size"
    )
    query_file_parser.add_argument(
        "--debug-profiles",
        action="store_true",
        help="Include the extracted query profile and top candidate profiles in the output",
    )

    subparsers.add_parser("health-check", help="Check local dependencies and configuration")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        from harbor_preindex.main import create_application

        app = create_application()

        if args.command == "build-index":
            payload = app.build_index(recreate=bool(args.recreate)).to_dict()
        elif args.command == "rescan":
            payload = app.build_index(recreate=True).to_dict()
        elif args.command == "query-file":
            if args.debug_profiles:
                payload = app.query_file_debug_payload(Path(args.file_path), top_k=args.top_k)
            else:
                payload = app.query_file(Path(args.file_path), top_k=args.top_k).to_dict()
        elif args.command == "health-check":
            payload = app.health_check()
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        logging.getLogger(__name__).exception("cli_command_failed")
        print(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
