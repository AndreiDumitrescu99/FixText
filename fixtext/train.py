import argparse
import logging
import random
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import Tuple

from models.bert import ClassificationBert
from models.trainer_helpers import get_optimizer, save_checkpoint, get_scheduler
from fixtext.utils.utils import set_seed
from fixtext.utils.argument_parser import get_my_args
from fixtext.utils.average_meter import AverageMeter
from fixtext.data.get_datasets import get_data_loaders
from fixtext.test import test

import logging

logger = logging.getLogger(__name__)

global_step = 0
best_loss = np.inf


def create_model() -> ClassificationBert:
    """
    Function that inits a Classification Bert Model.

    Returns:
        (ClassificationBert): The Classification Bert Model.
    """

    model = ClassificationBert()

    logger.info(
        "Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6)
    )

    return model


def main():
    """
    Main training function for Fix Text.
    """

    # Build the Argument Parser.
    parser = argparse.ArgumentParser(description="PyTorch FixText Training")

    # Extract the arguments.
    get_my_args(parser)
    args = parser.parse_args()

    global best_loss

    # Set the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))

    # Set Seed.
    if args.seed != -1:
        set_seed(args.seed)

    # Build Summary Writer.
    writer = SummaryWriter(args.out)

    # Get Data Loaders.
    (
        labeled_trainloader,
        unlabeled_trainloader,
        valid_loader,
        test_loader,
    ) = get_data_loaders(args)

    # Create the Classification BERT Model.
    model = create_model()
    model.to(args.device)

    # Get the optimizer.
    optimizer = get_optimizer(model, args)

    args.iteration = args.batch_size
    args.total_steps = args.epochs * args.iteration // args.gradient_accumulation_steps

    # Get the scheduler.
    scheduler = get_scheduler(optimizer, args)

    start_epoch = 0

    # Log info.
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.task}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    # Iterate through the epochs.
    for epoch in range(start_epoch, args.epochs):
        model.zero_grad()

        # Run a training epoch.
        train_loss, train_loss_x, train_loss_u, mask_prob = generic_train(
            args,
            labeled_trainloader,
            unlabeled_trainloader,
            model,
            optimizer,
            scheduler,
            epoch,
        )

        # Log details regarding the average losses.
        if args.no_progress:
            logger.info(
                "Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}.".format(
                    epoch + 1, train_loss, train_loss_x, train_loss_u
                )
            )

        test_model = model

        # Evaluate the model using: the training set, the validation set & the testing set.
        etrain_loss, bin_etrain = test(labeled_trainloader, test_model, args.device)
        valid_loss, bin_valid = test(valid_loader, test_model, args.device)
        real_test_loss, real_bin_test = test(test_loader, test_model, args.device)

        # Log the results.
        logger.info(f"Train metrics: {etrain_loss}")
        for k, v in bin_etrain.items():
            logger.info(f"{k}: {v}")
        logger.info(f"Valid metrics: {valid_loss}")
        for k, v in bin_valid.items():
            logger.info(f"{k}: {v}")
        logger.info(f"Test metrics: {real_test_loss}")
        for k, v in real_bin_test.items():
            logger.info(f"{k}: {v}")

        # Run the scheduler.
        scheduler.step(etrain_loss)

        # Log the evaluation result.
        writer.add_scalar("train/1.train_loss", train_loss, epoch)
        writer.add_scalar("train/2.train_loss_x", train_loss_x, epoch)
        writer.add_scalar("train/3.train_loss_u", train_loss_u, epoch)
        writer.add_scalar("train/4.mask", mask_prob, epoch)
        writer.add_scalar("test/2.test_loss", valid_loss, epoch)
        writer.add_scalar("eval_train/2.etrain_loss", etrain_loss, epoch)

        for k, v in bin_valid.items():
            writer.add_scalar(f"test/{k}", v, epoch)
        for k, v in bin_etrain.items():
            writer.add_scalar(f"eval_train/{k}", v, epoch)

        # Check if it is the best validation loss.
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)

        model_to_save = model.module if hasattr(model, "module") else model

        # Save the checkpoint.
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model_to_save.state_dict(),
                "test_loss": valid_loss,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.out,
        )

    writer.close()


def generic_train(
    args: argparse.Namespace,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    model: ClassificationBert,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
) -> Tuple[float, float, float, float]:
    """
    Generic train function.

    Args:
        args (argparse.Namespace): Object that contains hyperparameters needed for training.
        labeled_loader (DataLoader): Data Loader for the labeled training set.
        unlabeled_loader (DataLoader): Data Loader for the unlabeled training set.
        model (ClassificationBert): Model that is trained.
        optimizer (Optimizer): Optimizer used for training.
        scheduler (ReduceLROnPlateau): Scheduler used for training.
        epoch (int): Current epoch number.

    Returns:
        Tuple[float, float, float, float]:
            overall loss average, labeled loss average, unlabeled loss average, mask average
    """

    train_loss, train_loss_x, train_loss_u, mask_prob = train(
        args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, epoch
    )

    return train_loss, train_loss_x, train_loss_u, mask_prob


def train(
    args: argparse.Namespace,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    model: ClassificationBert,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
) -> Tuple[float, float, float, float]:
    """
    Runs a training step equivalent to a training epoch.

    Args:
        args (argparse.Namespace): Object that contains hyperparameters needed for training.
        labeled_loader (DataLoader): Data Loader for the labeled training set.
        unlabeled_loader (DataLoader): Data Loader for the unlabeled training set.
        model (ClassificationBert): Model that is trained.
        optimizer (Optimizer): Optimizer used for training.
        scheduler (ReduceLROnPlateau): Scheduler used for training.
        epoch (int): Current epoch number.

    Returns:
        Tuple[float, float, float, float]:
            overall loss average, labeled loss average, unlabeled loss average, mask average
    """

    # Initialize Average Meters.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    global global_step

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    # Build the train dataset loader by merging the labeled and unlabeled subsets.
    train_loader = zip(labeled_loader, unlabeled_loader)

    # Set the model in the training mode.
    model.train()

    if args.linear_lu:
        logger.info("Lu weight: ", (epoch / args.epochs) * args.lambda_u)

    # Iterate through the batches from the dataset.
    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        # Extract the labeled input text and its corresponding target.
        text_x, tgt_x, _ = data_x
        # Extract the unlabeled input text and its correponding augmented version.
        text_u, _, text_aug_data_u = data_u

        # With a given probability use the original text as the augmented version of it.
        if (text_aug_data_u is not None) and (random.random() < args.text_prob_aug):
            text_aug_u = text_aug_data_u
        else:
            text_aug_u = text_u

        data_time.update(time.time() - end)
        batch_size = text_x.shape[0]

        # Concatenate all the 3 texts.
        texts = torch.cat((text_x, text_u, text_aug_u)).to(args.device)

        # Cast the targets to the correct device.
        targets_x = tgt_x.to(args.device)

        # Run a prediction through the model.
        logits = model(texts)

        # Extract the logits for the labeled part.
        logits_x = logits[:batch_size]

        # Extract the logits for the unlabeled sample and its augmented version.
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        # Compute the Cross Entropy Loss Function for the labeled samples.
        Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

        # Extract pseudolabels for the original unlabeled sample.
        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        # Compute the Cross Entropy Loss Function between the logits corresponding to the augmented version
        # and the pseudolabels previously extracted.
        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask).mean()

        # If we use a random selection strategy for weighting the unlabeled loss, extract at random the weight.
        if args.random_lu:
            args.lambda_u = random.randint(args.lambda_u_min, args.lambda_u_max)

        # Weight the unlabeled loss, either by using a linear scheduler or directly.
        if args.linear_lu:
            Lu = (args.lambda_u_min + epoch * args.lambda_u / args.epochs) * Lu
        else:
            Lu = args.lambda_u * Lu

        # Compute the overall loss.
        loss = (Lx + Lu) / args.gradient_accumulation_steps

        # Backpropagate the loss.
        loss.backward()

        # Update the Meters.
        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())

        # If we accumulated enough steps, run the optimizer.
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        # Log details regarding the training iteration.
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

    # Return the losses averages.
    return losses.avg, losses_x.avg, losses_u.avg, mask_prob


if __name__ == "__main__":
    main()
