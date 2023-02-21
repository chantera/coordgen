import random
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

from coordgen import Span

from common.data import Sentence, Tree


class SpanSelector:
    def __call__(self, sentence: Sentence, num: int = 1) -> List[Span]:
        raise NotImplementedError


class RandomConstituentSelector(SpanSelector):
    def __init__(
        self,
        exclude_root: bool = False,
        filter: Optional[Callable[[Tree, Tuple[int, int]], bool]] = None,
    ):
        self.exclude_root = exclude_root
        self.filter = filter

    def __call__(self, sentence: Sentence, num: int = 1) -> List[Span]:
        if not sentence.tree:
            raise ValueError("sentence must have a parse tree")

        tokens = list(sentence.tree.leaves())
        length = len(tokens)

        spans = []
        for node, (start, end) in sentence.tree.traverse():
            if self.exclude_root and end - start == length:
                continue
            if self.filter and not self.filter(node, (start, end)):
                continue
            spans.append((start, end))

        if not spans:
            return spans

        random.shuffle(spans)
        spans = spans[:num]

        positions = list(to_char_positions(tokens, sentence.raw))
        spans = [(positions[s][0], positions[e - 1][1]) for (s, e) in spans]

        return spans


def to_char_positions(tokens: Iterable[str], raw: str) -> Iterator[Span]:
    offset = 0
    for token in tokens:
        idx = raw.find(token, offset)
        if idx < 0:
            raise KeyError(token)
        offset = idx + len(token)
        yield (idx, offset)
