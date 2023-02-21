import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple, Union


@dataclass
class Sentence:
    raw: str
    tree: Optional["Tree"] = None


@dataclass
class Tree:
    label: str
    children: List[Union["Tree", str]]

    def __contains__(self, node):
        return node in self.children

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_string(cls, s: str) -> "Tree":
        return next(parse_trees(s, check_exact_one=True))

    @property
    def is_preterminal(self):
        return len(self) == 1 and isinstance(self.children[0], str)

    def leaves(self) -> Iterator[str]:
        for child in self:
            if isinstance(child, Tree):
                yield from child.leaves()
            else:
                yield child

    def traverse(self) -> Iterator[Tuple["Tree", Tuple[int, int]]]:
        index = 0

        def _traverse(tree):
            nonlocal index
            begin = index
            for child in tree:
                if isinstance(child, str):
                    index += 1
                    break
                yield from _traverse(child)
            yield (tree, (begin, index))

        yield from _traverse(self)


_TREE_ESCAPE = {"(": "-LRB-", ")": "-RRB-", "{": "-LCB-", "}": "-RCB-", "[": "-LSB-", "]": "-RSB-"}
_TREE_UNESCAPE = {v: k for k, v in _TREE_ESCAPE.items()}


def _compile_token_regex(brackets: str) -> re.Pattern:
    assert len(brackets) == 2
    open_p = re.escape(brackets[0])
    close_p = re.escape(brackets[1])
    label_p = rf"[^\s{open_p}{close_p}]+"
    leaf_p = rf"[^\s{open_p}{close_p}]+"
    token_p = rf"{open_p}\s*({label_p})?|{close_p}|({leaf_p})"
    return re.compile(token_p)


_B_OPEN = "("
_B_CLOSE = ")"
_TOKEN_RE = _compile_token_regex(_B_OPEN + _B_CLOSE)


def parse_trees(text: Union[Iterable[str], str], check_exact_one: bool = False) -> Iterator[Tree]:
    context: List[str] = []
    stack: List[Tree] = []
    last_tree = None
    iter_idx = -1
    for s in [text] if isinstance(text, str) else text:
        iter_idx += 1
        context.append(s)
        for match in _TOKEN_RE.finditer(s):
            if check_exact_one and last_tree:
                _parse_error(iter_idx, context, match, "<end-of-string>")
            token = match.group()
            if token[0] == _B_OPEN:
                label = token[1:].lstrip()
                if not label and stack:
                    _parse_error(iter_idx, context, match, "<label>")
                stack.append(Tree(label, []))
            elif token == _B_CLOSE:
                if not stack:
                    _parse_error(iter_idx, context, match, _B_OPEN)
                node = stack.pop()
                if not node.children:
                    _parse_error(iter_idx, context, match, "<children>")
                elif len(node.children) > 1 and any(isinstance(c, str) for c in node.children):
                    _parse_error(iter_idx, context, match, "<single-terminal>")
                if stack:
                    stack[-1].children.append(node)
                    continue
                if last_tree:
                    yield last_tree
                last_tree = node
                context = context[-2:]  # keep at most 2 items
            else:
                if not stack:
                    _parse_error(iter_idx, context, match, _B_OPEN)
                stack[-1].children.append(_TREE_UNESCAPE.get(token, token))

    if stack:
        _parse_error(iter_idx, context, None, _B_CLOSE)
    elif check_exact_one and not last_tree:
        _parse_error(iter_idx, context, None, "<tree expression>")

    if last_tree:
        yield last_tree


class ParseError(ValueError):
    pass


def _parse_error(index: int, context: List[str], match: Union[re.Match, None], expected: str):
    token = match.group() if match else "<end-of-string>"
    position = match.start() if match else len(context[-1])
    message = f"expected {expected!r} but got {token!r} at ({index}, {position})"
    message += "\n{}".format("\n".join(c.rstrip() for c in context))
    raise ParseError(message)


def tree_to_str(tree: Tree) -> str:
    body = " ".join(
        str(child) if isinstance(child, Tree) else _TREE_ESCAPE.get(child, child) for child in tree
    )
    return f"{_B_OPEN}{tree.label} {body}{_B_CLOSE}"
