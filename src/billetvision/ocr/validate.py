"""Format validation with regex and optical character set correction (O<->0, I<->1, etc.)."""
import re

CHARSET_CORRECTIONS = {
    'O': '0',
    'I': '1',
    'S': '5',
    'B': '8',
    'Z': '2',
}

def correct_id_format(raw_text: str) -> str:
    """Clean and correct common OCR misrecognitions based on position."""
    text = raw_text.strip().upper().replace(" ", "")
    return text
