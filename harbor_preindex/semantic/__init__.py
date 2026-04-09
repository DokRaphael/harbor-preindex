"""Semantic enrichment layer."""

from harbor_preindex.semantic.base import SemanticEnricher
from harbor_preindex.semantic.code import CODE_EXTENSIONS, CodeSemanticEnricher, is_code_like
from harbor_preindex.semantic.document import DocumentSemanticEnricher
from harbor_preindex.semantic.models import EnrichedSignal, SemanticHints
from harbor_preindex.semantic.registry import SemanticEnricherRegistry

__all__ = [
    "CODE_EXTENSIONS",
    "CodeSemanticEnricher",
    "DocumentSemanticEnricher",
    "EnrichedSignal",
    "SemanticEnricher",
    "SemanticEnricherRegistry",
    "SemanticHints",
    "is_code_like",
]
