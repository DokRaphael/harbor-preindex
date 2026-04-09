"""Application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True, frozen=True)
class Settings:
    """Runtime settings."""

    app_name: str
    orchestrator_node: str
    log_level: str
    harbor_root: Path
    harbor_data_dir: Path
    results_dir: Path
    logs_dir: Path
    sqlite_path: Path
    qdrant_mode: str
    qdrant_path: Path
    qdrant_collection: str
    qdrant_file_collection: str
    top_k: int
    sample_files_per_directory: int
    max_text_snippet_chars: int
    max_profile_chars: int
    supported_extensions: tuple[str, ...]
    excluded_path_segments: tuple[str, ...]
    ollama_base_url: str
    ollama_api_key: str | None
    embedding_model: str
    llm_model: str
    ollama_timeout_seconds: float
    ollama_max_retries: int
    embedding_batch_size: int
    auto_accept_score: float
    auto_accept_score_gap: float
    llm_max_candidates: int

    def ensure_runtime_directories(self) -> None:
        """Create local runtime directories."""

        self.harbor_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load settings from .env and process environment variables."""

    load_dotenv(override=False)
    harbor_data_dir = _resolve_path(_read_str("HARBOR_DATA_DIR", "./data/runtime"))

    settings = Settings(
        app_name=_read_str("APP_NAME", "harbor-preindex"),
        orchestrator_node=_read_str("ORCHESTRATOR_NODE", "localhost"),
        log_level=_read_str("LOG_LEVEL", "INFO"),
        harbor_root=_resolve_path(_read_str("HARBOR_ROOT", "./data/storage-root")),
        harbor_data_dir=harbor_data_dir,
        results_dir=_resolve_path(_read_str("RESULTS_DIR", str(harbor_data_dir / "results"))),
        logs_dir=_resolve_path(_read_str("HARBOR_LOG_DIR", str(harbor_data_dir / "logs"))),
        sqlite_path=_resolve_path(
            _read_str("SQLITE_PATH", str(harbor_data_dir / "harbor-preindex.sqlite3"))
        ),
        qdrant_mode=_read_qdrant_mode(_read_str("QDRANT_MODE", "local")),
        qdrant_path=_resolve_path(_read_str("QDRANT_PATH", str(harbor_data_dir / "qdrant"))),
        qdrant_collection=_read_str("QDRANT_COLLECTION", "projects"),
        qdrant_file_collection=_read_str("QDRANT_FILE_COLLECTION", "files"),
        top_k=_read_int("TOP_K", 5),
        sample_files_per_directory=_read_int("SAMPLE_FILES_PER_DIRECTORY", 5),
        max_text_snippet_chars=_read_int("MAX_TEXT_SNIPPET_CHARS", 1200),
        max_profile_chars=_read_int("MAX_PROFILE_CHARS", 4000),
        supported_extensions=_read_extensions(
            os.getenv("SUPPORTED_EXTENSIONS"), (".txt", ".md", ".pdf")
        ),
        excluded_path_segments=_read_csv_lower_tuple(
            os.getenv("EXCLUDED_PATH_SEGMENTS"),
            (
                "build",
                "dist",
                "node_modules",
                ".git",
                "__pycache__",
                "example",
                "examples",
                "test",
                "tests",
                "testdata",
                "assets",
                "static",
                "libraries",
            ),
        ),
        ollama_base_url=_normalize_base_url(_read_str("OLLAMA_BASE_URL", "http://localhost:11434")),
        ollama_api_key=_read_optional_str("OLLAMA_API_KEY", default=None),
        embedding_model=_read_str("EMBEDDING_MODEL", "embeddinggemma"),
        llm_model=_read_str("LLM_MODEL", "qwen2.5:7b-instruct"),
        ollama_timeout_seconds=_read_float("OLLAMA_TIMEOUT_SECONDS", 60.0),
        ollama_max_retries=_read_non_negative_int("OLLAMA_MAX_RETRIES", 2),
        embedding_batch_size=_read_positive_int("EMBEDDING_BATCH_SIZE", 16),
        auto_accept_score=_read_float("AUTO_ACCEPT_SCORE", 0.90),
        auto_accept_score_gap=_read_float("AUTO_ACCEPT_SCORE_GAP", 0.12),
        llm_max_candidates=_read_int("LLM_MAX_CANDIDATES", 5),
    )
    settings.ensure_runtime_directories()
    return settings


def _read_str(env_key: str, default: str) -> str:
    value = os.getenv(env_key)
    if value is not None and value.strip():
        return value.strip()
    return default


def _read_optional_str(env_key: str, default: str | None = None) -> str | None:
    value = os.getenv(env_key)
    if value is not None:
        cleaned = value.strip()
        return cleaned or default
    return default


def _read_int(env_key: str, default: int) -> int:
    value = os.getenv(env_key)
    if value is not None and value.strip():
        return int(value)
    return default


def _read_positive_int(env_key: str, default: int) -> int:
    value = _read_int(env_key, default)
    if value <= 0:
        raise ValueError(f"{env_key} must be greater than zero")
    return value


def _read_non_negative_int(env_key: str, default: int) -> int:
    value = _read_int(env_key, default)
    if value < 0:
        raise ValueError(f"{env_key} must be zero or greater")
    return value


def _read_float(env_key: str, default: float) -> float:
    value = os.getenv(env_key)
    if value is not None and value.strip():
        return float(value)
    return default


def _read_extensions(env_value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    return _read_csv_lower_tuple(env_value, default)


def _read_csv_lower_tuple(env_value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if env_value:
        values = [item.strip().lower() for item in env_value.split(",") if item.strip()]
        return tuple(values) if values else default
    return default


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _normalize_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4]
    return normalized


def _read_qdrant_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized != "local":
        raise ValueError(
            f"unsupported QDRANT_MODE={value!r}; only 'local' is supported in this MVP"
        )
    return normalized
