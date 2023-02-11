from typing import Iterable, List, Optional, Tuple

import torch
import transformers

from coordgen._core import Coord, CoordinationGenerator, Span
from coordgen.models._utils import embed_coord, embed_mask
from coordgen.models.generation_utils import SynchronizedLogitsProcessor


class BertForCoordinationGeneration(CoordinationGenerator):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.cc = "and"

    def generate(self, inputs: Iterable[Tuple[str, Span]]) -> List[Tuple[str, Coord]]:
        inputs = list(inputs)

        batch_append_inputs = []
        batch_prepend_inputs = []
        for raw, span in inputs:
            num_tokens = len(self.tokenizer.tokenize(raw[span[0] : span[1]]))
            mask = " ".join([self.tokenizer.mask_token] * num_tokens)
            s1, s2 = embed_mask(mask, raw, span, self.cc)
            batch_append_inputs.append(s1)
            batch_prepend_inputs.append(s2)

        decoding = self._forward(batch_append_inputs, batch_prepend_inputs)

        outputs = []
        for ids, (raw, span) in zip(decoding, inputs):
            text = self.tokenizer.decode(ids).strip()
            s = embed_coord(text, raw, span, self.cc)
            outputs.append(s)

        return outputs

    @torch.no_grad()
    def _forward(self, inputs1: List[str], inputs2: List[str]) -> List[List[int]]:
        assert len(inputs1) == len(inputs2)
        batch = self.tokenizer(inputs1, inputs2, padding=True, return_tensors="pt")
        if self.device:
            batch = batch.to(self.device)

        self.model.eval()
        logits = self.model(**batch).logits

        outputs = []
        logits_processor = SynchronizedLogitsProcessor()
        for i, ids in enumerate(batch.input_ids):
            (start1, end1), (start2, end2) = _find_span_pair(
                ids, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id
            )
            scores = logits_processor.forward(logits[i, start1:end1], logits[i, start2:end2])
            outputs.append(scores.argmax(dim=-1).tolist())

        return outputs


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
