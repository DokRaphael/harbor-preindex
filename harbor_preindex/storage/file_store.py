"""Local JSON result persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harbor_preindex.schemas import BatchQueryResult, IndexBuildSummary, QueryResult, RetrievalResponse
from harbor_preindex.utils.text import slugify, utc_now_compact


class JsonResultStore:
    """Persist stable JSON artifacts locally."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_index_summary(self, summary: IndexBuildSummary) -> Path:
        filename = f"index_{utc_now_compact()}.json"
        return self._write(filename, summary.to_dict())

    def save_query_result(self, result: QueryResult) -> Path:
        stem = slugify(Path(result.input_file).stem)
        filename = f"query_{utc_now_compact()}_{stem}.json"
        return self._write(filename, result.to_dict())

    def save_batch_query_result(self, result: BatchQueryResult) -> Path:
        stem = slugify(Path(result.input_path).name or "batch")
        filename = f"batch_query_{utc_now_compact()}_{stem}.json"
        return self._write(filename, result.to_dict())

    def save_retrieval_response(self, response: RetrievalResponse) -> Path:
        stem = slugify(response.query)
        filename = f"retrieval_{utc_now_compact()}_{stem}.json"
        return self._write(filename, response.to_dict())

    def _write(self, filename: str, payload: dict[str, Any]) -> Path:
        output_path = self.base_dir / filename
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path
