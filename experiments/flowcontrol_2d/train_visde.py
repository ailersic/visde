import torch
from torch.utils.data import DataLoader

import pickle as pkl
import os
import pathlib

import pytorch_lightning as pl
from pytorch_lightning import loggers

import visde
from experiments.flowcontrol_2d.def_model import create_latent_sde

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def get_dataloaders(n_win: int,
                    n_batch: int
) -> tuple[DataLoader, DataLoader]:
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)

    train_data = visde.MultiEvenlySpacedTensors(data["train_mu"], data["train_t"], data["train_x"], data["train_f"], n_win)
    val_data = visde.MultiEvenlySpacedTensors(data["val_mu"], data["val_t"], data["val_x"], data["val_f"], n_win)

    train_sampler = visde.MultiTemporalSampler(train_data, n_batch, n_repeats=1)
    train_dataloader = DataLoader(
        train_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=train_sampler,
        pin_memory=True,
    )
    val_sampler = visde.MultiTemporalSampler(val_data, n_batch, n_repeats=1)
    val_dataloader = DataLoader(
        val_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=val_sampler,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader

def main():
    dim_z = 9
    n_win = 1
    n_batch = 64

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch)

    device = torch.device("cuda:0")
    model = create_latent_sde(dim_z, n_batch, n_win)

    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_visde")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=500,
        logger=tensorboard,
        check_val_every_n_epoch=50
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()