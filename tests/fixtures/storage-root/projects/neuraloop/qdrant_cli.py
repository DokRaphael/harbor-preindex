"""Command-line entrypoint for Qdrant maintenance."""

import argparse

from qdrant_client import QdrantClient


class QdrantSyncService:
    """Synchronize vectors with the local Qdrant store."""

    def sync_points(self) -> None:
        client = QdrantClient(path="/tmp/qdrant")
        print(client)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdrant-sync")
    parser.add_argument("--collection", required=True)
    return parser
