import os
import torch
import shutil
import torch.optim as optim

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=args.lr_patience, verbose=True, factor=args.lr_factor, min_lr=0.000001
    )


def get_optimizer(model, args):

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer