from utils.average_meter import AverageMeter
import torch
import torch.nn.functional as F
import tqdm
import time
from utils.metrics import binary_metrics

def test(args, test_loader, model, tst_name='Test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    bin_metrics = {m : AverageMeter() for m in [
        'micro/precision', 'micro/recall', 'micro/f1',
        'macro/precision', 'macro/recall', 'macro/f1',
    ]}
    
    if isinstance(test_loader, dict):
        test_loader = test_loader['test']
        
    if not args.no_progress:
        test_loader = tqdm(test_loader)
        
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
            if not args.no_progress:
                test_loader.set_description(tst_name+" Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                ))
        if not args.no_progress:
            test_loader.close()


    for k in bin_metrics:
        bin_metrics[k] = bin_metrics[k].avg

    return losses.avg, bin_metrics