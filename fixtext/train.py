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
from utils.argument_parser import get_my_args
from data.get_datasets import get_data_loaders
from test import test

import logging

logger = logging.getLogger(__name__)

global_step = 0
best_loss = np.inf


def main():
    parser = argparse.ArgumentParser(description="PyTorch FixText Training")

    get_my_args(parser)
    args = parser.parse_args()

    global best_loss

    def create_model():
        model = ClassificationBert()

        logger.info(
            "Total params: {:.2f}M".format(
                sum(p.numel() for p in model.parameters()) / 1e6
            )
        )

        return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    print("Num GPUs", args.n_gpu)
    print("Device", args.device)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")

    logger.info(dict(args._get_kwargs()))

    if args.seed != -1:
        set_seed(args.seed)

    writer = SummaryWriter(args.out)

    (
        labeled_trainloader,
        unlabeled_trainloader,
        valid_loader,
        test_loader,
    ) = get_data_loaders(args)

    model = create_model()
    model.to(args.device)

    optimizer = get_optimizer(model, args)
    # scaler = torch.cuda.amp.GradScaler()

    args.iteration = args.batch_size
    args.total_steps = args.epochs * args.iteration // args.gradient_accumulation_steps

    scheduler = get_scheduler(optimizer, args)

    start_epoch = 0

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.task}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    if args.eval_only:
        checkpoint = torch.load(
            "C:\\Users\\andre\\Desktop\\Master - 2\\Research\\experiments_fixtext\\short_set\\twitter_classic\\model_best.pth.tar"
        )
        model.load_state_dict(checkpoint["state_dict"])
        test_model = model

        valid_loss, bin_valid = test(args, valid_loader, test_model, "Valid")
        print("Valid metrics: ", valid_loss)
        for k, v in bin_valid.items():
            print("", k, v, sep="\t")
        print()
        test_loss, bin_test = test(args, test_loader, test_model, "Test")
        print("Test metrics: ", test_loss)
        for k, v in bin_test.items():
            print("", k, v, sep="\t")

        return

    for epoch in range(start_epoch, args.epochs):
        model.zero_grad()

        # Train step
        train_loss, train_loss_x, train_loss_u, mask_prob = generic_train(
            args,
            labeled_trainloader,
            unlabeled_trainloader,
            model,
            optimizer,
            scheduler,
            epoch,
        )

        if args.no_progress:
            logger.info(
                "Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}.".format(
                    epoch + 1, train_loss, train_loss_x, train_loss_u
                )
            )

        test_model = model

        # Test step
        etrain_loss, bin_etrain = test(
            args, labeled_trainloader, test_model, "EvalTrain"
        )
        print("Train metrics: ", etrain_loss)
        for k, v in bin_etrain.items():
            print("", k, v, sep="\t")
        print()
        test_loss, bin_test = test(args, valid_loader, test_model, "Valid")
        print("Valid metrics: ", test_loss)
        for k, v in bin_test.items():
            print("", k, v, sep="\t")
        print()
        real_test_loss, real_bin_test = test(args, test_loader, test_model, "Test")
        print("Test metrics: ", real_test_loss)
        for k, v in real_bin_test.items():
            print("", k, v, sep="\t")
        print()

        tuning_metric = etrain_loss
        scheduler.step(tuning_metric)

        writer.add_scalar("train/1.train_loss", train_loss, epoch)
        writer.add_scalar("train/2.train_loss_x", train_loss_x, epoch)
        writer.add_scalar("train/3.train_loss_u", train_loss_u, epoch)
        writer.add_scalar("train/4.mask", mask_prob, epoch)
        writer.add_scalar("test/2.test_loss", test_loss, epoch)
        writer.add_scalar("eval_train/2.etrain_loss", etrain_loss, epoch)

        for k, v in bin_test.items():
            writer.add_scalar(f"test/{k}", v, epoch)
        for k, v in bin_etrain.items():
            writer.add_scalar(f"eval_train/{k}", v, epoch)

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        model_to_save = model.module if hasattr(model, "module") else model

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model_to_save.state_dict(),
                "test_loss": test_loss,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.out,
        )

    writer.close()


def generic_train(
    args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch
):
    train_loss, train_loss_x, train_loss_u, mask_prob = train(
        args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch
    )

    return train_loss, train_loss_x, train_loss_u, mask_prob


def train(args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    global global_step

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(labeled_loader, unlabeled_loader)
    model.train()
    if args.linear_lu:
        print("Lu weight: ", (epoch / args.epochs) * args.lambda_u)

    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        text_x, tgt_x = data_x
        text_u, _, text_aug_data_u = data_u

        if (text_aug_data_u is not None) and (random.random() < args.text_prob_aug):
            text_aug_u = text_aug_data_u
        else:
            text_aug_u = text_u

        data_time.update(time.time() - end)
        batch_size = text_x.shape[0]

        texts = torch.cat((text_x, text_u, text_aug_u)).to(args.device)

        targets_x = tgt_x.to(args.device)

        logits = model(texts)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        # Compute Loss
        Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")
        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask).mean()

        if args.random_lu:
            args.lambda_u = random.randint(args.lambda_u_min, args.lambda_u_max)

        if args.linear_lu:
            Lu = (args.lambda_u_min + epoch * args.lambda_u / args.epochs) * Lu
        else:
            Lu = args.lambda_u * Lu

        loss = (Lx + Lu) / args.gradient_accumulation_steps

        loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())

        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        mask_prob = mask.mean().item()
        if not args.no_progress:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_prob,
                )
            )
            p_bar.update()

    if not args.no_progress:
        p_bar.close()

    return losses.avg, losses_x.avg, losses_u.avg, mask_prob


if __name__ == "__main__":
    main()
