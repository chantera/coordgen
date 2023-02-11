from typing import List

import torch
import transformers


class GenerationMixin(transformers.GenerationMixin):
    def generate(self, *args, **kwargs):
        self._synchronized = kwargs.pop("synchronize", False)
        self._bad_token_ids = kwargs.pop("bad_token_ids", None)
        return super().generate(*args, **kwargs)

    def _get_logits_processor(self, *args, **kwargs):
        processors = super()._get_logits_processor(*args, **kwargs)
        if self._synchronized:
            processors.append(SynchronizedLogitsProcessor())
        if self._bad_token_ids is not None:
            processors.append(NoBadTokenLogitsProcessor(list(self._bad_token_ids)))
        return processors


class SynchronizedLogitsProcessor(transformers.LogitsProcessor):
    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits1, logits2 = logits.split(logits.size(0) // 2)
        logits = self.forward(logits1, logits2)
        return torch.cat((logits, logits), dim=0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x1, x2)


class NoBadTokenLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, bad_token_ids: List[int]):
        self.bad_token_ids = bad_token_ids

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits[:, self.bad_token_ids] = -float("inf")
        return logits
