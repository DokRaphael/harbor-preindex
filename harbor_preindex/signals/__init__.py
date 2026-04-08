"""Signal extraction abstractions."""

from harbor_preindex.signals.base import SignalExtractor
from harbor_preindex.signals.document import DocumentSignalExtractor
from harbor_preindex.signals.models import ExtractedSignal
from harbor_preindex.signals.registry import SignalExtractorRegistry

__all__ = [
    "DocumentSignalExtractor",
    "ExtractedSignal",
    "SignalExtractor",
    "SignalExtractorRegistry",
]
