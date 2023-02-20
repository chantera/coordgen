import random
from os import PathLike
from typing import Optional, Set, Tuple, Union

import torch
from coordgen.models import AutoModelForCoordinationGeneration

from common.data import Sentence, Tree, parse_trees
from common.selectors import RandomConstituentSelector


def generate(
    input_file: Union[str, bytes, PathLike],
    model_name_or_path: str,
    num_spans: int = 1,
    batch_size: int = 20,
    cuda: bool = False,
    seed: Optional[int] = None,
):
    CC_TOKENS = {"and", "or", "but", "nor", "and/or"}
    TARGET_LABELS = {"NP", "VP", "ADJP", "ADVP", "PP", "S", "SBAR"}
    STOP_WORDS: Set[str] = set()
    MIN_SENTENCE_LENGTH = 10

    def _filter(node: Tree, span: Tuple[int, int]) -> bool:
        if node.label not in TARGET_LABELS:
            return False
        if node.is_preterminal and next(node.leaves()).lower() in STOP_WORDS:
            return False
        return True

    if seed is not None:
        random.seed(seed)

    sentences = []
    with open(input_file) as f:
        for tree in parse_trees(f):
            tokens = list(tree.leaves())
            if len(tokens) < MIN_SENTENCE_LENGTH:
                continue
            if any(token.lower() in CC_TOKENS for token in tokens):
                continue
            sentences.append(Sentence(" ".join(tokens), tree))

    model = AutoModelForCoordinationGeneration.from_pretrained(
        model_name_or_path, device=torch.device("cuda" if cuda else "cpu")
    )
    selector = RandomConstituentSelector(exclude_root=False, filter=_filter)

    inputs = [(s, span) for s in sentences for span in selector(s, num_spans)]
    for offset in range(0, len(inputs), batch_size):
        batch = inputs[offset : offset + batch_size]
        results = model.generate((s.raw, span) for s, span in batch)
        for raw, coord in results:
            pre, post = coord.conjuncts
            print(
                "{}[{}]{}[{}]{}".format(
                    raw[: pre[0]],
                    raw[pre[0] : pre[1]],
                    raw[pre[1] : post[0]],
                    raw[post[0] : post[1]],
                    raw[post[1] :],
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    generate(
        args.input_file,
        args.model,
        args.num,
        args.batch_size,
        args.cuda,
        args.seed,
    )
