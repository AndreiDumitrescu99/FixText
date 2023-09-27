from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from typing import Dict


def binary_metrics(output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Given a prediction and a target, compute multiple binary metrics such as: precision, recall or f1.

    Args:
        output (torch.Tensor): Output prediction from the trained model.
        target (torch.Tensor): Target for the given output.

    Returns:
        (Dict[str, float]): A dictionary with metrics and the scores obtained.
    """

    output = output.argmax(dim=1).cpu()
    target = target.cpu()

    binary_metrics = {
        "micro/precision": precision_score(
            target, output, average="micro", zero_division=0
        ),
        "micro/recall": recall_score(target, output, average="micro", zero_division=0),
        "micro/f1": f1_score(target, output, average="micro", zero_division=0),
        "macro/precision": precision_score(
            target, output, average="macro", zero_division=0
        ),
        "macro/recall": recall_score(target, output, average="macro", zero_division=0),
        "macro/f1": f1_score(target, output, average="macro", zero_division=0),
        "accuracy": accuracy_score(target, output),
    }

    return binary_metrics
