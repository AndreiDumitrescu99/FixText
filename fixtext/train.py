from utils.average_meter import AverageMeter
import argparse
import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.bert import ClassificationBert
from models.trainer_helpers import get_optimizer, save_checkpoint, get_scheduler
from utils.utils import set_seed
from data.get_datasets import get_data_loaders
from .test import test

import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixText Training')

    get_my_args(parser)
    args = parser.parse_args()

    global best_loss
    best_loss = np.inf

    def create_model(args):
        model = ClassificationBert()

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))

        return model

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    print("Num GPUs", args.n_gpu)
    print("Device", args.device)
    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
    )

    logger.info(dict(args._get_kwargs()))

    if args.seed != -1:
        set_seed(args)

    writer = SummaryWriter(args.out)

    labeled_trainloader, unlabeled_trainloader, valid_loader, test_loader = get_data_loaders(args)

    model = create_model(args)
    model.to(args.device)
    
    optimizer = get_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler()

    args.iteration = args.batch_size
    args.total_steps = args.epochs * args.iteration // args.gradient_accumulation_steps
    
    scheduler = get_scheduler(optimizer, args)

    start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        print('GPU: ', args.gpu_id)

        if os.path.isfile(args.resume):
            args.out = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)

            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            print(f'Model loaded from epoch {start_epoch}')

            model.to(args.device)
            print('Model sent to device')

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print('Attention! No checkpoint directory found. Training from epoch 0.')

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    
    if args.eval_only:

        test_model = model
            
        valid_loss, bin_valid = test(args, valid_loader, test_model, 'Valid')
        print('Valid metrics: ', valid_loss)
        for k, v in bin_valid.items():
            print('', k, v, sep='\t')
        print()
        test_loss, bin_test = test(args, test_loader, test_model, 'Test')
        print('Test metrics: ', test_loss)
        for k, v in bin_test.items():
            print('', k, v, sep='\t')


        return

    for epoch in range(start_epoch, args.epochs):
        model.zero_grad()

        # Train step
        train_loss, train_loss_x, train_loss_u, mask_prob = generic_train(
                args, labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, epoch, scaler)

        if args.no_progress:
            logger.info("Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}."
                        .format(epoch+1, train_loss, train_loss_x, train_loss_u))


        test_model = model

        # Test step
        etrain_loss, bin_etrain = test(args, labeled_trainloader, test_model, 'EvalTrain')
        test_loss, bin_test = test(args, valid_loader, test_model, 'Valid')

        tuning_metric = etrain_loss
        scheduler.step(tuning_metric)
            
        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('train/2.train_loss_x', train_loss_x, epoch)
        writer.add_scalar('train/3.train_loss_u', train_loss_u, epoch)
        writer.add_scalar('train/4.mask', mask_prob, epoch)
        writer.add_scalar('test/2.test_loss', test_loss, epoch)
        writer.add_scalar('eval_train/2.etrain_loss', etrain_loss, epoch)

        for k, v in bin_test.items():
            writer.add_scalar(f'test/{k}', v, epoch)
        for k, v in bin_etrain.items():
            writer.add_scalar(f'eval_train/{k}', v, epoch)

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        model_to_save = model.module if hasattr(model, "module") else model

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'test_loss': test_loss,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

    writer.close()

        
def generic_train(args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch, scaler):
    train_loss, train_loss_x, train_loss_u, mask_prob = train(
        args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch, scaler)

    return train_loss, train_loss_x, train_loss_u, mask_prob
        

def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

"""
args.no_progress
args.distil
"""
def train(args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    global global_step
    
    if not args.no_progress:
        p_bar = tqdm(range(args.iteration),
                     disable=args.local_rank not in [-1, 0])
        
    train_loader = zip(labeled_loader, unlabeled_loader)
    model.train()
    if args.linear_lu or args.distil =='linear':
        print('Lu weight: ', (epoch / args.epochs) * args.lambda_u)

    print('args.distil', args.distil)

    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        text_x, segment_x, tgt_x, text_aug_data_x = data_x
        text_u, segment_u, _, text_aug_data_u = data_u

        if text_aug_data_x is not None:
            raise NotImplementedError('treat this case')

        if (text_aug_data_u is not None) and (random.random() < args.text_prob_aug):
            text_aug_u, segment_aug_u = text_aug_data_u
            if batch_idx < 4:
                print('use aug text')
        else:
            text_aug_u, segment_aug_u = text_u, segment_u

        data_time.update(time.time() - end)
        batch_size = text_x.shape[0]
        
        texts = torch.cat((text_x, text_u, text_aug_u)).to(args.device)
        segments = torch.cat((segment_x, segment_u, segment_aug_u)).to(args.device)

        targets_x = tgt_x.to(args.device)

        logits = model(texts, segments)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        ### STILL TO DO ###
        if args.distil == 'unlabeled':
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = softXEnt(logits_u_s, pseudo_label)
        
        elif args.distil == 'linear':
#             raise ValueError('unimplemented')
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu_soft = softXEnt(logits_u_s, pseudo_label)
            Lu_hard = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            alpha = 1 - epoch / args.epochs
            Lu = alpha * Lu_soft + (1-alpha) * Lu_hard
        
        else:
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        if args.random_lu:
            args.lambda_u = random.randint(args.lambda_u_min, args.lambda_u_max)
            
        if args.linear_lu:
#             Lu = (epoch / args.epochs) * Lu
            Lu = (args.lambda_u_min + epoch * args.lambda_u / args.epochs) * Lu
        else:
            Lu = args.lambda_u * Lu
        
        loss = (Lx + Lu) / args.gradient_accumulation_steps
    #         loss = Lx + Lu

        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()            
            if args.scheduler == 'cosine':
                scheduler.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        mask_prob = mask.mean().item()
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=optimizer.param_groups[0]['lr'],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_prob))
            p_bar.update()

    if not args.no_progress:
        p_bar.close()

    return losses.avg, losses_x.avg, losses_u.avg, mask_prob
