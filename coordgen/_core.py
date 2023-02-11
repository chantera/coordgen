from dataclasses import dataclass
from typing import Iterable, List, Tuple

Span = Tuple[int, int]  # (start, end]


@dataclass
class Coord:
    cc: Span
    conjuncts: List[Span]


class CoordinationGenerator:
    def generate(self, inputs: Iterable[Tuple[str, Span]]) -> List[Tuple[str, Coord]]:
        raise NotImplementedError
