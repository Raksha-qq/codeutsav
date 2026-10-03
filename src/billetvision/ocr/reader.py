"""OCR reader interface supporting PaddleOCR / EasyOCR fallbacks."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class OcrResult:
    text: str
    confidence: float
    source: str = "ocr" # "ocr", "qr", "barcode", "manual"
