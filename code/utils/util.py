# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import torch


def load_checkpoint(path, model, optimizer, from_ddp=False):
    """loads previous checkpoint

    Args:
        path (str): path to checkpoint
        model (model): model to restore checkpoint to
        optimizer (optimizer): torch optimizer to load optimizer state_dict to
        from_ddp (bool, optional): load DistributedDataParallel checkpoint to regular model. Defaults to False.

    Returns:
        model, optimizer, epoch_num, loss
    """
    # load checkpoint
    checkpoint = torch.load(path)
    # transfer state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])
    # transfer optimizer state_dict from checkpoint to model
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # track loss
    loss = checkpoint["loss"]
    return model, optimizer, checkpoint["epoch"], loss.item()


def restore_model(logger, snapshot_path, model_num=None):
    """wrapper function to read log dir and load restore a previous checkpoint

    Args:
        logger (Logger): logger object (for info output to console)
        snapshot_path (str): path to checkpoint directory

    Returns:
        model, optimizer, start_epoch, performance
    """
    try:
        # check if there is previous progress to be restored:
        logger.info(f"Snapshot path: {snapshot_path}")
        iter_num = []
        name = "model_iter"
        if model_num:
            name = model_num
        for filename in os.listdir(snapshot_path):
            if name in filename:
                basename, extension = os.path.splitext(filename)
                iter_num.append(int(basename.split("_")[2]))
        iter_num = max(iter_num)
        for filename in os.listdir(snapshot_path):
            if name in filename and str(iter_num) in filename:
                model_checkpoint = filename
    except Exception as e:
        logger.warning(f"Error finding previous checkpoints: {e}")

    try:
        logger.info(f"Restoring model checkpoint: {model_checkpoint}")
        model, optimizer, start_epoch, performance = load_checkpoint(
            snapshot_path + "/" + model_checkpoint, model, optimizer
        )
        logger.info(f"Models restored from iteration {iter_num}")
        return model, optimizer, start_epoch, performance
    except Exception as e:
        logger.warning(f"Unable to restore model checkpoint: {e}, using new model")


def save_checkpoint(epoch, model, optimizer, loss, path):
    """Saves model as checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


class Logger:
    """Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), "wb") as fp:
            pickle.dump(self.data, fp, -1)
