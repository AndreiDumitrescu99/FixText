from utils.average_meter import AverageMeter
import torch
import torch.nn.functional as F
import tqdm
import time
from utils.metrics import binary_metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def test(args, test_loader, model, tst_name="Test"):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)

            loss = F.cross_entropy(outputs, targets)

            losses.update(loss.item(), inputs.shape[0])

            b_metrics = binary_metrics(outputs, targets)

            for metric, value in b_metrics.items():
                bin_metrics[metric].update(value, inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            outputs = outputs.argmax(dim=1).cpu()
            targets = targets.cpu()

            if full_target is None:
                full_target = targets
                full_pred = outputs
            else:
                full_pred = torch.cat((full_pred, outputs))
                full_target = torch.cat((full_target, targets))

    print(
        "TRUE F1-SCORE: ",
        f1_score(full_target, full_pred, average="macro", zero_division=0),
    )
    for k in bin_metrics:
        bin_metrics[k] = bin_metrics[k].avg

    return losses.avg, bin_metrics
