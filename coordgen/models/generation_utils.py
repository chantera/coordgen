import torch
import transformers


class SynchronizedLogitsProcessor(transformers.LogitsProcessor):
    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits1, logits2 = logits.split(logits.size(0) // 2)
        logits = self.forward(logits1, logits2)
        return torch.cat((logits, logits), dim=0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x1, x2)
