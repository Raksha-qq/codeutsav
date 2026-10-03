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
    """Clean and correct common OCR misrecognitions (e.g. 'H12345O' -> 'H123450')."""
    text = raw_text.strip().upper().replace(" ", "")
    if len(text) > 1 and text[0].isalpha():
        # First char is heat prefix letter; subsequent characters should be digits
        prefix = text[0]
        suffix = "".join(CHARSET_CORRECTIONS.get(c, c) for c in text[1:])
        return prefix + suffix
    return text

def validate_heat_id(text: str, pattern: str = r"^[A-Z]\d{5,7}$") -> bool:
    """Validate heat ID string against configured regex pattern."""
    return bool(re.match(pattern, text))

