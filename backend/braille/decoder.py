"""Convert detected cell patterns into English text."""

from __future__ import annotations

from dataclasses import dataclass

from braille.patterns import DIGIT_PATTERNS, PATTERN_TO_CHAR


@dataclass
class BrailleCell:
    pattern: int
    confidence: float = 1.0
    row: int = 0
    col: int = 0


def cells_to_text(cells: list[BrailleCell]) -> str:
    out: list[str] = []
    number_mode = False
    capitalize_next = False

    for cell in cells:
        p = cell.pattern
        ch = PATTERN_TO_CHAR.get(p)

        if ch == "#":
            number_mode = True
            continue
        if ch == "CAP":
            capitalize_next = True
            continue

        if number_mode and p in DIGIT_PATTERNS:
            ch = DIGIT_PATTERNS[p]
        elif ch is None:
            ch = "?"

        if capitalize_next and ch and ch.isalpha():
            ch = ch.upper()
            capitalize_next = False

        if number_mode and ch and not (ch.isdigit() or ch in ".,-"):
            number_mode = False

        out.append(ch)

    return "".join(out)
