from argparse import Namespace
from models.bert import ClassificationBert
from sklearn.metrics import f1_score
import time
from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.metrics import binary_metrics
from utils.average_meter import AverageMeter


def test(
    test_loader: DataLoader, model: ClassificationBert, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Main testing function for Fix Text.

    Args:
        test_loader (DataLoader): Data Loader with which we test the model.
        model (ClassificationBert): Model to be evaluated.
        device (torch.device): The device on which we keep the model & tensors.

    Returns:
        Tuple[float, Dict[str, float]]:
            Loss average on the dataset and a dictionary with the metrics of interest and the obtained averages..
    """

    # Build Average Meters for batch time, data time and losses.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Build dictionary with Average Meters for the binary metrics.
    bin_metrics = {
        m: AverageMeter()
        for m in [
            "micro/precision",
            "micro/recall",
            "micro/f1",
            "macro/precision",
            "macro/recall",
            "macro/f1",
            "accuracy",
        ]
    }

    if isinstance(test_loader, dict):
        test_loader = test_loader["test"]

    end = time.time()

    full_target = None
    full_pred = None

    # Run through the dataset.
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            # Set the model in evaluation mode.
            model.eval()

            # Cast the inputs and targets to the given device.
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Run the input through the model.
            outputs = model(inputs)

            # Compute the Cross Entropy Loss Function between the outputs and targets.
            loss = F.cross_entropy(outputs, targets)
            losses.update(loss.item(), inputs.shape[0])

            # Compute the binary metrics
            b_metrics = binary_metrics(outputs, targets)
            for metric, value in b_metrics.items():
                bin_metrics[metric].update(value, inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            outputs = outputs.argmax(dim=1).cpu()
            targets = targets.cpu()

            # Save the targets and outputs to compute correctly the Macro and Micro F1 Scores.
            if full_target is None:
                full_target = targets
                full_pred = outputs
            else:
                full_pred = torch.cat((full_pred, outputs))
                full_target = torch.cat((full_target, targets))

    # Compute the Macro and Micro F1 Scores.
    macro_f1_score = (
        f1_score(full_target, full_pred, average="macro", zero_division=0),
    )
    micro_f1_score = (
        f1_score(full_target, full_pred, average="micro", zero_division=0),
    )

    # Store the values.
    bin_metrics["macro/f1"].avg = macro_f1_score
    bin_metrics["micro/f1"].avg = micro_f1_score

    # Save only the averages.
    for k in bin_metrics:
        bin_metrics[k] = bin_metrics[k].avg

    return losses.avg, bin_metrics
