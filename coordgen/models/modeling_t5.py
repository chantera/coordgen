from typing import Iterable, List, Optional, Tuple

import torch
import transformers

from coordgen._core import Coord, CoordinationGenerator, Span
from coordgen.models._utils import embed_coord, embed_mask
from coordgen.models.generation_utils import GenerationMixin


class T5ForCoordinationGeneration(CoordinationGenerator):
    EXTRA_TOKEN_0 = "<extra_id_0>"
    EXTRA_TOKEN_1 = "<extra_id_1>"

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ):
        if not isinstance(model, GenerationMixin):
            mixin_name = f"{GenerationMixin.__module__}.{GenerationMixin.__qualname__}"
            raise TypeError(f"model must implement `{mixin_name}`")

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.cc = "and"
        self.num_beams = 4

        allowed_tokens = [self.EXTRA_TOKEN_0, self.EXTRA_TOKEN_1]
        allowed_token_ids = set(tokenizer.convert_tokens_to_ids(allowed_tokens))
        self._all_special_ids = set(self.tokenizer.all_special_ids)
        self._bad_token_ids = self._all_special_ids - allowed_token_ids

    def generate(self, inputs: Iterable[Tuple[str, Span]]) -> List[Tuple[str, Coord]]:
        inputs = list(inputs)

        batch_append_inputs = []
        batch_prepend_inputs = []
        for raw, span in inputs:
            s1, s2 = embed_mask(self.EXTRA_TOKEN_0, raw, span, self.cc)
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
        batch = self.tokenizer(inputs1 + inputs2, padding=True, return_tensors="pt")
        if self.device:
            batch = batch.to(self.device)

        extra_token_0_id = self.tokenizer.convert_tokens_to_ids(self.EXTRA_TOKEN_0)
        extra_token_1_id = self.tokenizer.convert_tokens_to_ids(self.EXTRA_TOKEN_1)
        offset = 3  # <pad> <extra_id_0> [...] <extra_id_1>

        self.model.eval()
        decoding = self.model.generate(
            **batch,
            min_length=offset,
            max_length=offset + self.tokenizer.model_max_length,
            early_stopping=True,
            num_beams=self.num_beams,
            num_return_sequences=1,
            bos_token_id=extra_token_0_id,
            eos_token_id=extra_token_1_id,
            forced_bos_token_id=extra_token_0_id,
            forced_eos_token_id=extra_token_1_id,
            bad_token_ids=self._bad_token_ids,
            synchronize=True,
        )

        outputs = []
        max_length = decoding.size(1)
        for ids in decoding[: len(decoding) // 2].tolist():
            i = j = 2  # skip "<pad>" and "<extra_id_0>"
            while j < max_length and ids[j] not in self._all_special_ids:
                j += 1
            outputs.append(ids[i:j])

        return outputs


class T5ForConditionalGeneration(transformers.T5ForConditionalGeneration, GenerationMixin):
    pass
