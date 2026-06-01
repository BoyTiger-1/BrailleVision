"""Grade 1 English Braille: 6-dot pattern (bits 1,2,4,8,16,32) to character."""

# Dot positions: 1 4 / 2 5 / 3 6 (top-left numbering)
PATTERN_TO_CHAR: dict[int, str] = {
    0x01: "a",
    0x03: "b",
    0x09: "c",
    0x19: "d",
    0x11: "e",
    0x0B: "f",
    0x1B: "g",
    0x13: "h",
    0x0D: "i",
    0x1D: "j",
    0x05: "k",
    0x07: "l",
    0x0F: "m",
    0x17: "n",
    0x15: "o",
    0x0E: "p",
    0x1E: "q",
    0x16: "r",
    0x1C: "s",
    0x14: "t",
    0x2D: "u",
    0x27: "v",
    0x3A: "w",
    0x2E: "x",
    0x3E: "y",
    0x34: "z",
    0x00: " ",
    # Number sign prefix uses # — handled in decoder
    0x3F: "#",  # number indicator when alone
    0x06: "CAP",  # capital indicator (dots 4-6) — simplified
    0x04: "'",  # apostrophe / single quote
    0x02: ",",
    0x12: "-",
    0x10: ".",
    0x08: "?",
    0x18: "!",
    0x28: "(",
    0x38: ")",
    0x20: '"',
    0x3C: "/",
    0x30: ";",
    0x24: ":",
}

# A–J dot patterns also represent digits 1–0 when preceded by number sign
DIGIT_PATTERNS = {
    0x01: "1",
    0x03: "2",
    0x09: "3",
    0x19: "4",
    0x11: "5",
    0x0B: "6",
    0x1B: "7",
    0x13: "8",
    0x0D: "9",
    0x1D: "0",
}

CHAR_TO_PATTERN: dict[str, int] = {v: k for k, v in PATTERN_TO_CHAR.items() if v not in ("CAP",)}


def pattern_to_unicode(pattern: int) -> str:
    return chr(0x2800 + pattern)


def pattern_bits_from_dots(active: list[int]) -> int:
    bits = 0
    for d in active:
        if 1 <= d <= 6:
            bits |= 1 << (d - 1)
    return bits
