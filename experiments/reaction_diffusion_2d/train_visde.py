import torch
from torch.utils.data import DataLoader

import pickle as pkl
import os
import pathlib

import pytorch_lightning as pl
from pytorch_lightning import loggers

import visde
from experiments.reaction_diffusion_2d.def_model import create_latent_sde

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"

def get_dataloaders(n_win: int,
                    n_batch: int
) -> tuple[DataLoader, DataLoader]:
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)

    train_data = visde.MultiEvenlySpacedTensors(data["train_mu"], data["train_t"], data["train_x"], data["train_f"], n_win)
    val_data = visde.MultiEvenlySpacedTensors(data["val_mu"], data["val_t"], data["val_x"], data["val_f"], n_win)

    train_sampler = visde.MultiTemporalSampler(train_data, n_batch, n_repeats=1)
    train_dataloader = DataLoader(
        train_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=train_sampler,
        pin_memory=True
    )
    val_sampler = visde.MultiTemporalSampler(val_data, n_batch, n_repeats=1)
    val_dataloader = DataLoader(
        val_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=val_sampler,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def main():
    dim_z = 2
    n_win = 1
    n_batch = 128
    print(torch.cuda.is_available())

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch)
    model = create_latent_sde(dim_z, n_batch, n_win, DATA_FILE, device)

    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_visde")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=1000,
        logger=tensorboard,
        check_val_every_n_epoch=200
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()