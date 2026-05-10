import torch
from torch import nn


def normalized_cross_entropy_loss(logits: torch.Tensor, gt_labels: torch.Tensor, num_classes: int):
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(logits, gt_labels.long()).mean()
    loss /= torch.log(torch.tensor(num_classes, dtype=torch.float32, device=logits.device))
    return loss
