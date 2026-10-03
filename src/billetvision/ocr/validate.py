"""OCR text validation with regex matching and position-aware charset correction.

Charset corrections compensate for common OCR confusion between visually similar
characters (e.g., letter O vs digit 0, letter I vs digit 1).

Correction is position-aware: when a regex defines expected character classes per
position, we apply the right direction of correction (letter→digit or digit→letter).
"""
from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Characters that look like digits but are read as letters
LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
    "T": "7",
    "L": "1",
    "Q": "0",
}

# Reverse: digits that look like letters
DIGIT_TO_LETTER: dict[str, str] = {v: k for k, v in LETTER_TO_DIGIT.items()}
# Keep only unambiguous reversals
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
}


def _apply_map(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def _regex_expects_digit_at(pattern: str, pos: int) -> Optional[bool]:
    """Naively check if the compiled regex atom at ``pos`` wants a digit or letter.

    Returns True  → digit expected
            False → letter expected
            None  → unknown / complex pattern
    """
    # Only handle simple patterns like ^[A-Z]\\d{5,7}$
    try:
        compiled = re.compile(pattern)
        # Use fullmatch against a probe string to classify each position
        alpha_probe = "AAAAAAAAA"
        digit_probe = "999999999"
        # Try to see if position pos must be a digit
        prefix_len = pos
        # Build a string of the right length, vary position pos
        if compiled.fullmatch(alpha_probe[:prefix_len] + "0" + alpha_probe[prefix_len + 1 :]):
            return True
        if compiled.fullmatch(digit_probe[:prefix_len] + "A" + digit_probe[prefix_len + 1 :]):
            return False
    except Exception:
        pass
    return None


def correct_for_digits(text: str) -> str:
    """Apply letter→digit corrections to the whole string."""
    return _apply_map(text, LETTER_TO_DIGIT)


def correct_for_letters(text: str) -> str:
    """Apply digit→letter corrections to the whole string."""
    return _apply_map(text, DIGIT_TO_LETTER)


def correct_position_aware(text: str, pattern: str) -> str:
    """Apply corrections guided by what each regex position expects.

    For each character in ``text``:
    - If the regex expects a digit at this position and the char looks like a letter → correct
    - If the regex expects a letter at this position and the char looks like a digit → correct
    - Otherwise → leave as-is
    """
    result = list(text)
    for i, ch in enumerate(text):
        expects_digit = _regex_expects_digit_at(pattern, i)
        if expects_digit is True and ch in LETTER_TO_DIGIT:
            result[i] = LETTER_TO_DIGIT[ch]
        elif expects_digit is False and ch in DIGIT_TO_LETTER:
            result[i] = DIGIT_TO_LETTER[ch]
    return "".join(result)


def normalize(text: str) -> str:
    """Strip, uppercase, remove internal whitespace."""
    return text.strip().upper().replace(" ", "").replace("-", "").replace("_", "")


def validate_regex(text: str, pattern: str) -> bool:
    """Return True if ``text`` matches ``pattern`` exactly (fullmatch)."""
    try:
        return bool(re.fullmatch(pattern, text))
    except re.error:
        logger.warning("Invalid regex pattern: %s", pattern)
        return False


def correct_and_validate(
    raw_text: str,
    pattern: str,
) -> Tuple[str, bool, float]:
    """Attempt to correct ``raw_text`` until it matches ``pattern``.

    Tries in order:
    1. Normalized raw text
    2. Position-aware correction guided by regex
    3. All letter→digit corrections
    4. All digit→letter corrections
    5. Letter→digit on normalized text

    Args:
        raw_text: Raw OCR output string.
        pattern: Regex pattern (fullmatch) for a valid billet ID.

    Returns:
        (corrected_text, matched, confidence_penalty)
        confidence_penalty: 0.0 if raw matched, 0.05 per correction step applied.
    """
    norm = normalize(raw_text)

    # Step 1: raw normalized
    if validate_regex(norm, pattern):
        return norm, True, 0.0

    # Step 2: position-aware
    pa = correct_position_aware(norm, pattern)
    if validate_regex(pa, pattern):
        return pa, True, 0.05

    # Step 3: all letter→digit
    l2d = correct_for_digits(norm)
    if validate_regex(l2d, pattern):
        return l2d, True, 0.10

    # Step 4: all digit→letter
    d2l = correct_for_letters(norm)
    if validate_regex(d2l, pattern):
        return d2l, True, 0.10

    # Step 5: position-aware on letter→digit result
    pa2 = correct_position_aware(l2d, pattern)
    if validate_regex(pa2, pattern):
        return pa2, True, 0.15

    # Give up — return normalized raw, unmatched
    return norm, False, 0.20


# Legacy alias used by existing test_scaffold.py
def correct_id_format(raw_text: str) -> str:
    """Normalize and apply letter→digit corrections (backward-compatible wrapper)."""
    return correct_for_digits(normalize(raw_text))


def validate_heat_id(text: str, pattern: str = r"^[A-Z]\d{5,7}$") -> bool:
    """Validate a heat ID string against the default (or given) regex."""
    return validate_regex(text, pattern)
