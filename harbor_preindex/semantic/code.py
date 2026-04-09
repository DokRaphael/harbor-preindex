"""Semantic enrichment for code-like text files."""

from __future__ import annotations

import re
from pathlib import Path

from harbor_preindex.semantic.base import SemanticEnricher
from harbor_preindex.semantic.models import EnrichedSignal, SemanticHints
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.utils.text import normalize_text

CODE_EXTENSIONS = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

_TOPIC_ALIASES = {
    "api": "api_client",
    "argparse": "cli",
    "cli": "cli",
    "click": "cli",
    "client": "api_client",
    "config": "configuration",
    "dotenv": "configuration",
    "embed": "embeddings",
    "embedding": "embeddings",
    "extract": "text_extraction",
    "httpx": "api_client",
    "json": "serialization",
    "ollama": "embeddings",
    "parser": "parsing",
    "pdf": "pdf_processing",
    "pypdf": "pdf_processing",
    "pytest": "tests",
    "qdrant": "vector_storage",
    "query": "retrieval",
    "request": "api_client",
    "retriev": "retrieval",
    "search": "retrieval",
    "settings": "configuration",
    "sqlite": "storage",
    "storage": "storage",
    "test": "tests",
    "toml": "serialization",
    "typer": "cli",
    "unittest": "tests",
    "xml": "serialization",
    "yaml": "serialization",
}


class CodeSemanticEnricher(SemanticEnricher):
    """Build compact semantic hints for code and config-like text files."""

    def supports(self, file_path: Path, signal: ExtractedSignal) -> bool:
        excerpt = str(signal.metadata.get("text_excerpt", ""))
        return is_code_like(file_path, excerpt)

    def enrich(self, file_path: Path, signal: ExtractedSignal) -> EnrichedSignal:
        excerpt = normalize_text(str(signal.metadata.get("text_excerpt", "")))
        language_hint = _language_hint(file_path.suffix.lower())
        imports = _extract_imports(excerpt, language_hint)
        symbols = _extract_symbols(excerpt, language_hint)
        doc_hint = _extract_doc_hint(excerpt)

        entity_candidates = _compact_list(imports, limit=5)
        topic_hints = _topic_hints(
            file_path=file_path,
            imports=imports,
            symbols=symbols,
            excerpt=excerpt,
        )
        structure_hints = _structure_hints(
            file_path=file_path,
            imports=imports,
            symbols=symbols,
            excerpt=excerpt,
        )
        kind_hints = _kind_hints(file_path=file_path, imports=imports, excerpt=excerpt)
        evidence_hints = _compact_list([doc_hint, *imports[:3], *symbols[:3]], limit=6)
        functional_summary = _functional_summary(
            language_hint=language_hint,
            kind_hints=kind_hints,
            topic_hints=topic_hints,
            symbols=symbols,
        )

        hints = SemanticHints(
            kind_hints=kind_hints,
            topic_hints=topic_hints,
            entity_candidates=entity_candidates,
            time_hints=[],
            structure_hints=structure_hints,
            language_hint=language_hint,
            functional_summary=functional_summary,
            evidence_hints=evidence_hints,
        )
        metadata = {
            **signal.metadata,
            "semantic_hints": hints.to_dict(),
            "imports": imports,
            "symbols": symbols,
        }
        return EnrichedSignal(
            modality="code",
            semantic_hints=hints,
            metadata=metadata,
            confidence=min(0.99, signal.confidence + 0.02),
        )


def is_code_like(file_path: Path, excerpt: str) -> bool:
    suffix = file_path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return True

    code_markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "function ",
        "const ",
        "#include",
        "SELECT ",
        "BEGIN ",
        "public class ",
    )
    return any(marker in excerpt for marker in code_markers)


def _language_hint(suffix: str) -> str:
    mapping = {
        ".bash": "shell",
        ".cfg": "config",
        ".conf": "config",
        ".css": "css",
        ".go": "go",
        ".html": "html",
        ".ini": "ini",
        ".java": "java",
        ".js": "javascript",
        ".json": "json",
        ".jsx": "javascript",
        ".php": "php",
        ".py": "python",
        ".rb": "ruby",
        ".rs": "rust",
        ".sh": "shell",
        ".sql": "sql",
        ".toml": "toml",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".zsh": "shell",
    }
    return mapping.get(suffix, suffix.lstrip(".") or "code")


def _extract_imports(excerpt: str, language_hint: str) -> list[str]:
    if language_hint == "python":
        matches = re.findall(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)", excerpt, re.MULTILINE)
        return _compact_list([match.split(".")[0] for match in matches], limit=8)

    patterns = [
        r"^\s*import\s+([A-Za-z0-9_@/\.\-]+)",
        r"^\s*from\s+([A-Za-z0-9_@/\.\-]+)",
        r"require\([\"']([^\"']+)[\"']\)",
        r"^\s*using\s+([A-Za-z0-9_\.]+)",
        r"^\s*#include\s+[<\"]([^>\"]+)[>\"]",
    ]
    imports: list[str] = []
    for pattern in patterns:
        imports.extend(re.findall(pattern, excerpt, re.MULTILINE))
    return _compact_list([value.split("/")[0].split(".")[0] for value in imports], limit=8)


def _extract_symbols(excerpt: str, language_hint: str) -> list[str]:
    if language_hint == "python":
        symbols = re.findall(
            r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
            excerpt,
            re.MULTILINE,
        )
        return _compact_list(symbols, limit=8)

    patterns = [
        r"^\s*(?:function|class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(",
        r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)",
    ]
    symbols: list[str] = []
    for pattern in patterns:
        symbols.extend(re.findall(pattern, excerpt, re.MULTILINE))
    return _compact_list(symbols, limit=8)


def _extract_doc_hint(excerpt: str) -> str | None:
    lines = [line.strip(" #/\"'") for line in excerpt.splitlines() if line.strip()]
    for line in lines[:4]:
        if len(line) >= 12 and not _looks_like_code_line(line):
            return line
    return None


def _topic_hints(
    file_path: Path,
    imports: list[str],
    symbols: list[str],
    excerpt: str,
) -> list[str]:
    tokens = [file_path.stem, *imports, *symbols, *re.findall(r"[A-Za-z_]{4,}", excerpt)]
    topics: list[str] = []
    for token in tokens:
        lowered = token.lower()
        for alias, topic in _TOPIC_ALIASES.items():
            if alias in lowered:
                topics.append(topic)
        if lowered in {"qdrant", "qdrant_client", "pdf", "pypdf", "cli"}:
            topics.append(lowered.replace("_client", ""))
    return _compact_list(topics, limit=6)


def _structure_hints(
    file_path: Path,
    imports: list[str],
    symbols: list[str],
    excerpt: str,
) -> list[str]:
    hints: list[str] = []
    if imports:
        hints.append("uses_imports")
    if symbols:
        hints.append("defines_symbols")
    if re.search(r"^\s*class\s+", excerpt, re.MULTILINE):
        hints.append("defines_classes")
    if re.search(r"^\s*(?:async\s+def|def|function|func)\s+", excerpt, re.MULTILINE):
        hints.append("defines_functions")
    if excerpt.startswith("#!") or excerpt.startswith("```"):
        hints.append("script_entrypoint")
    if "test" in file_path.name.lower():
        hints.append("test_surface")
    return _compact_list(hints, limit=6)


def _kind_hints(file_path: Path, imports: list[str], excerpt: str) -> list[str]:
    lowered_name = file_path.name.lower()
    hints = ["code_artifact"]
    if "test" in lowered_name:
        hints.append("test_module")
    elif file_path.suffix.lower() in {".sh", ".bash", ".zsh"} or excerpt.startswith("#!"):
        hints.append("automation_script")
    elif file_path.suffix.lower() in {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".xml"}:
        hints.append("configuration_artifact")
    else:
        hints.append("code_module")
    if any(
        import_name in {"qdrant", "qdrant_client", "sqlite", "httpx"}
        for import_name in imports
    ):
        hints.append("integration_module")
    return _compact_list(hints, limit=4)


def _functional_summary(
    language_hint: str,
    kind_hints: list[str],
    topic_hints: list[str],
    symbols: list[str],
) -> str:
    base = f"{language_hint.capitalize()} {kind_hints[-1].replace('_', ' ')}"
    parts: list[str] = [base]
    if topic_hints:
        parts.append("covering " + ", ".join(topic_hints[:3]).replace("_", " "))
    if symbols:
        parts.append("with symbols " + ", ".join(symbols[:3]))
    return ". ".join(parts) + "."


def _looks_like_code_line(value: str) -> bool:
    return bool(re.search(r"[{}();=]|^\s*(?:from|import|def|class|return)\b", value))


def _compact_list(values: list[str | None], limit: int) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip().strip(",;:")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compacted.append(cleaned)
        if len(compacted) >= limit:
            break
    return compacted
