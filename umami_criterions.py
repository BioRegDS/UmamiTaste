import torch
import torch.nn.functional as F


class PosWeightedBCEWithLogitsLoss(torch.nn.Module):
    """Binary BCE-with-logits with a scalar positive-class weight.

    The scalar is converted to a tensor on the same device/dtype as logits
    at each forward pass, which makes it easy to configure from JSON.
    """

    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("pos_weight must be > 0")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.pos_weight = float(pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        pw = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pw,
            reduction=self.reduction,
        )
