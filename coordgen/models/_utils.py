from typing import Tuple

from coordgen._core import Coord, Span


def embed_mask(mask: str, raw: str, span: Span, cc: str) -> Tuple[str, str]:
    start, end = span
    # NOTE:
    # - `head` has a trailing space
    # - `body` has no leading or trailing space
    # - `tail` has a leading space
    head, body, tail = raw[:start], raw[start:end], raw[end:]
    cc = " " + cc + " "
    return (f"{head}{body}{cc}{mask}{tail}", f"{head}{mask}{cc}{body}{tail}")


def embed_coord(text: str, raw: str, span: Span, cc: str) -> Tuple[str, Coord]:
    start, end = span
    cc = " " + cc + " "
    start2 = end + len(cc)
    end2 = start2 + len(text)
    coord = Coord(cc=(end + 1, start2 - 1), conjuncts=[(start, end), (start2, end2)])
    return f"{raw[:end]}{cc}{text}{raw[end:]}", coord
