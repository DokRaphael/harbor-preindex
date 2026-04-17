"""CLI interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from harbor_preindex import __version__

FEEDBACK_REASON_CHOICES = (
    "correct_match",
    "wrong_path",
    "wrong_parent",
    "should_have_split",
    "should_not_have_split",
    "review_was_correct",
    "review_was_unnecessary",
    "bad_new_subfolder_proposal",
    "good_new_subfolder_proposal_bad_name",
    "ambiguous",
    "other",
)


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

    query_batch_parser = subparsers.add_parser(
        "query-batch",
        help="Build a placement plan for an incoming directory or single file",
    )
    query_batch_parser.add_argument("input_path", help="Path to the input directory or file")
    query_batch_parser.add_argument(
        "--top-k", type=int, default=None, help="Override top-k retrieval size"
    )
    query_batch_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect files directly inside the input directory",
    )
    query_batch_parser.add_argument(
        "--debug-profiles",
        action="store_true",
        help="Include per-file extracted profiles and candidate folder profiles in the output",
    )

    query_parser = subparsers.add_parser(
        "query", help="Search the retrieval core for files and folders"
    )
    query_parser.add_argument("text", help="Plain text retrieval query")
    query_parser.add_argument(
        "--top-k", type=int, default=None, help="Override top-k retrieval size"
    )
    query_parser.add_argument(
        "--debug-evidence",
        action="store_true",
        help="Include structured retrieval evidence in the output",
    )

    feedback_parser = subparsers.add_parser(
        "feedback", help="Record lightweight human feedback for a persisted result"
    )
    feedback_subparsers = feedback_parser.add_subparsers(dest="feedback_command")

    mark_good_parser = feedback_subparsers.add_parser(
        "mark-good", help="Record a good result"
    )
    mark_good_parser.add_argument("result_id", help="Persisted result identifier")
    mark_good_parser.add_argument(
        "--reason",
        choices=FEEDBACK_REASON_CHOICES,
        default="correct_match",
        help="Bounded reason for the feedback event",
    )
    mark_good_parser.add_argument(
        "--notes",
        default=None,
        help="Optional short human note",
    )

    mark_bad_parser = feedback_subparsers.add_parser("mark-bad", help="Record a bad result")
    mark_bad_parser.add_argument("result_id", help="Persisted result identifier")
    mark_bad_parser.add_argument(
        "--reason",
        choices=FEEDBACK_REASON_CHOICES,
        required=True,
        help="Bounded reason for the feedback event",
    )
    mark_bad_parser.add_argument(
        "--notes",
        default=None,
        help="Optional short human note",
    )

    correct_parser = feedback_subparsers.add_parser(
        "correct", help="Record a corrected destination for a result"
    )
    correct_parser.add_argument("result_id", help="Persisted result identifier")
    correct_parser.add_argument(
        "--path",
        required=True,
        help="Corrected destination path",
    )
    correct_parser.add_argument(
        "--parent-path",
        default=None,
        help="Corrected parent path when useful",
    )
    correct_parser.add_argument(
        "--reason",
        choices=FEEDBACK_REASON_CHOICES,
        required=True,
        help="Bounded reason for the feedback event",
    )
    correct_parser.add_argument(
        "--notes",
        default=None,
        help="Optional short human note",
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
        elif args.command == "query-batch":
            if args.debug_profiles:
                payload = app.query_batch_debug_payload(
                    Path(args.input_path),
                    top_k=args.top_k,
                    recursive=not args.no_recursive,
                )
            else:
                payload = app.query_batch(
                    Path(args.input_path),
                    top_k=args.top_k,
                    recursive=not args.no_recursive,
                ).to_dict()
        elif args.command == "query":
            payload = app.query(args.text, top_k=args.top_k).to_dict(
                include_evidence=bool(args.debug_evidence)
            )
        elif args.command == "feedback":
            if not args.feedback_command:
                parser.print_help()
                return 1
            if args.feedback_command == "mark-good":
                payload = app.record_feedback(
                    args.result_id,
                    feedback_status="good",
                    feedback_reason=args.reason,
                    notes=args.notes,
                ).to_dict()
            elif args.feedback_command == "mark-bad":
                payload = app.record_feedback(
                    args.result_id,
                    feedback_status="bad",
                    feedback_reason=args.reason,
                    notes=args.notes,
                ).to_dict()
            elif args.feedback_command == "correct":
                payload = app.record_feedback(
                    args.result_id,
                    feedback_status="corrected",
                    feedback_reason=args.reason,
                    corrected_path=args.path,
                    corrected_parent_path=args.parent_path,
                    notes=args.notes,
                ).to_dict()
            else:
                parser.print_help()
                return 1
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
