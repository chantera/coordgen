from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import transformers

Span = Tuple[int, int]  # (start, end]


@dataclass
class Coord:
    cc: Span
    conjuncts: List[Span]


class CoordinationGenerator:
    def generate(self, inputs: Iterable[Tuple[str, Span]]) -> List[Tuple[str, Coord]]:
        raise NotImplementedError


class BertForCoordinationGeneration(CoordinationGenerator):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cc = "and"

    def generate(self, inputs: Iterable[Tuple[str, Span]]) -> List[Tuple[str, Coord]]:
        inputs = list(inputs)
        outputs: List[Tuple[str, Coord]] = []

        batch_append_inputs = []
        batch_prepend_inputs = []
        for raw, span in inputs:
            s1, s2 = self.embed_mask(self.tokenizer, raw, span, self.cc)
            batch_append_inputs.append(s1)
            batch_prepend_inputs.append(s2)

        model_inputs = self.tokenizer(
            batch_append_inputs, batch_prepend_inputs, padding=True, return_tensors="pt"
        )
        if self.device:
            model_inputs = model_inputs.to(self.device)
        print(model_inputs.input_ids.shape)
        logits = self.model(**model_inputs).logits

        model_outputs = []
        logits_processor = SynchronizedLogitsProcessor()
        for i, ids in enumerate(model_inputs.input_ids):
            (start1, end1), (start2, end2) = _find_span_pair(
                ids, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id
            )
            scores = logits_processor.forward(logits[i, start1:end1], logits[i, start2:end2])
            model_outputs.append(scores.argmax(dim=-1))

        for ids, (raw, span) in zip(model_outputs, inputs):
            text = self.tokenizer.decode(ids.tolist()).strip()
            s = self.embed_coord(text, raw, span, self.cc)
            outputs.append(s)

        return outputs

    @staticmethod
    def embed_mask(tokenizer, raw: str, span: Span, cc: str) -> Tuple[str, str]:
        start, end = span
        # NOTE:
        # - `head` has a trailing space
        # - `body` has no leading or trailing space
        # - `tail` has a leading space
        head, body, tail = raw[:start], raw[start:end], raw[end:]
        cc = " " + cc + " "
        mask = " ".join([tokenizer.mask_token] * len(tokenizer.tokenize(body)))
        return (f"{head}{body}{cc}{mask}{tail}", f"{head}{mask}{cc}{body}{tail}")

    @staticmethod
    def embed_coord(text: str, raw: str, span: Span, cc: str) -> Tuple[str, Coord]:
        start, end = span
        cc = " " + cc + " "
        start2 = end + len(cc)
        end2 = start2 + len(text)
        coord = Coord(cc=(end + 1, start2 - 1), conjuncts=[(start, end), (start2, end2)])
        return f"{raw[:end]}{cc}{text}{raw[end:]}", coord


def _find_span_pair(ids, sep_token_id, mask_token_id):
    length = len(ids)

    sep_idx = 0
    while sep_idx < length and ids[sep_idx] != sep_token_id:
        sep_idx += 1
    assert sep_idx < length, "could not find sep_token_id"

    start1 = 0
    while start1 < sep_idx and ids[start1] != mask_token_id:
        start1 += 1
    assert start1 < sep_idx, "could not find mask_token_id"
    end1 = start1
    while end1 < sep_idx and ids[end1] == mask_token_id:
        end1 += 1

    start2 = sep_idx + 1
    while start2 < length and ids[start2] != mask_token_id:
        start2 += 1
    assert start2 < length, "could not find mask_token_id"
    end2 = start2
    while end2 < length and ids[end2] == mask_token_id:
        end2 += 1

    return (start1, end1), (start2, end2)


class SynchronizedLogitsProcessor(transformers.LogitsProcessor):
    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits1, logits2 = logits.split(logits.size(0) // 2)
        logits = self.forward(logits1, logits2)
        return torch.cat((logits, logits), dim=0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x1, x2)
