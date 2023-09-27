import os
import torch
import shutil
import torch.optim as optim
import argparse
from fixtext.models.bert import ClassificationBert
from typing import Dict


def save_checkpoint(
    state: Dict[str, any],
    is_best: bool,
    checkpoint: str,
    filename: str = "checkpoint.pth.tar",
) -> None:
    """
    Function that saves the a checkpoint for the Classification BERT.

    Args:
        state (Dict[str, any]): Object that stores the model state, the optimizer state, the scheduler
            state, the current epoch, the current validation loss and the best loss so far. This object
            represents the state that will be saved.
        is_best (bool): Boolean flag that tells us if this is the best checkpoint so far, in terms
            of validation loss. In this case the model will be saved as the "model_best.pth.tar".
            It will overwrite the old best checkpoint.
        checkpoint (str): Folder path where the checkpoints are stored.
        filename (str): Name of the checkpoint.

    Returns:
        (None)
    """

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def get_scheduler(
    optimizer: torch.optim.AdamW, args: argparse.Namespace
) -> optim.lr_scheduler.ReduceLROnPlateau:
    """
    Returns a ReduceLROnPlateau learning rate scheduler.

    Args:
        optimizer (torch.optim.AdamW): Optimizer used for the learning rate scheduler.
        args (argparse.Namespace): Object that stores the learning rate patience in
        the "lr_patience" property, and the learning rate factor in the "lr_factor"
        property for the scheduler.

    Returns:
        (optim.lr_scheduler.ReduceLROnPlateau): The learning rate scheduler.
    """

    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=args.lr_patience,
        verbose=True,
        factor=args.lr_factor,
        min_lr=0.000001,
    )


def get_optimizer(
    model: ClassificationBert, args: argparse.Namespace
) -> torch.optim.AdamW:
    """
    Returns an AdamW optimizer for the Classification BERT model. We set a custom learning rate for the
    BERT Layer and other learning rate for the Linear Layer.

    Args:
        model (ClassificationBert): Model for which we want to get the optimizer.
        args (argparse.Namespace): Object that stores the learning rate for the BERT Layer
            in the "lrmain" property, and the learning rate for the Linear Layer in "lrlast".

    Returns:
        (torch.optim.AdamW): The optimizer for the Classification BERT.
    """

    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.linear.parameters(), "lr": args.lrlast},
        ]
    )
    return optimizer
