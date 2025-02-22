import pytorch_lightning as pl
import torch
import math

from dataclasses import dataclass
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from jaxtyping import jaxtyped
from beartype import beartype

import os
import pickle as pkl
import numpy as np
import pathlib

from experiments.forced_burgers_1d.def_model import EncodeMeanNet, EncodeVarNet, DecodeMeanNet, VarDecoderWithLogNormalPrior

import visde
# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data_100_20_50.pkl"

@dataclass(frozen=True)
class TestAutoencoderConfig:
    n_totaldata: int
    n_samples: int
    n_tquad: int
    n_warmup: int
    n_transition: int
    lr: float
    lr_sched_freq: int

class TestAutoencoder(pl.LightningModule):
    """Variational inference of latent SDE"""

    _empty_tensor: Tensor  # empty tensor to get device
    _iter_count: int  # iteration count
    config: TestAutoencoderConfig
    encoder: visde.VarEncoder
    decoder: visde.VarDecoder

    def __init__(
        self,
        config: TestAutoencoderConfig,
        encoder: visde.VarEncoder,
        decoder: visde.VarDecoder,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer("_empty_tensor", torch.empty(0))
        self._iter_count = 0

    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    @jaxtyped(typechecker=beartype)
    def training_step(self,
                      batch: list[Tensor],
                      batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single training step"""

        self._iter_count += 1

        mu, t, x_win, x, f = batch

        # t.shape can either be (n_batch,) or (n_batch, 1); this ensures the latter
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # assess autoencoder
        z_win_mean, _ = self.encoder(mu.to(self.device), x_win.to(self.device))
        x_rec_mean, _ = self.decoder(mu.to(self.device), z_win_mean)
        raw_rmse = (x_rec_mean - x.to(self.device)).pow(2).mean().sqrt()
        norm_rmse = raw_rmse / (x.to(self.device).pow(2).mean().sqrt())

        self.log("train/raw_rmse", raw_rmse.item(), prog_bar=True)
        self.log("train/norm_rmse", norm_rmse.item(), prog_bar=True)

        return raw_rmse
    
    def validation_step(self,
                        batch: list[Tensor],
                        batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single validation step"""

        mu, t, x_win, x, f = batch

        # t.shape can either be (n_batch,) or (n_batch, 1); this ensures the latter
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        
        # assess autoencoder
        z_win_mean, _ = self.encoder(mu.to(self.device), x_win.to(self.device))
        x_rec_mean, _ = self.decoder(mu.to(self.device), z_win_mean)
        raw_rmse = (x_rec_mean - x.to(self.device)).pow(2).mean().sqrt()
        norm_rmse = raw_rmse / (x.to(self.device).pow(2).mean().sqrt())

        self.log("val/raw_rmse", raw_rmse.item(), prog_bar=True)
        self.log("val/norm_rmse", norm_rmse.item(), prog_bar=True)

        return raw_rmse

    def configure_optimizers(self):  # type: ignore
        # ------------------------------------------------------------------------------
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        gamma = math.exp(math.log(0.9) / self.config.lr_sched_freq)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "step",  # call scheduler after every train step
                "frequency": 1,
            },
        }

def get_dataloaders(n_win: int,
                    n_batch: int,
                    data_file: str
) -> tuple[DataLoader, DataLoader]:
    with open(os.path.join(CURR_DIR, data_file), "rb") as f:
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

def create_autoencoder(dim_z: int, n_win: int, data_file: str) -> tuple[TestAutoencoderConfig, visde.VarEncoder, visde.VarDecoder]:
    with open(os.path.join(CURR_DIR, data_file), "rb") as f:
        data = pkl.load(f)
    
    dim_mu = data["train_mu"].shape[1]
    shape_x = tuple(data["train_x"].shape[2:])
    dim_x = int(np.prod(shape_x))

    config = TestAutoencoderConfig(n_totaldata=torch.numel(data["train_t"]),
                                   n_samples=1,
                                   n_tquad=10,
                                   n_warmup=0,
                                   n_transition=5000,
                                   lr=1e-3,
                                   lr_sched_freq=1000)

    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu,
                                           dim_x=dim_x,
                                           dim_z=dim_z,
                                           shape_x=shape_x,
                                           n_win=n_win)

    # encoder
    encode_mean_net = EncodeMeanNet(vaeconfig)
    encode_var_net = EncodeVarNet(vaeconfig)
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)

    # decoder        
    decode_mean_net = DecodeMeanNet(vaeconfig)
    decoder = VarDecoderWithLogNormalPrior(vaeconfig, decode_mean_net)

    return config, encoder, decoder

def main():
    dim_z = 9
    n_win = 1
    n_batch = 128

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch, DATA_FILE)

    config, encoder, decoder = create_autoencoder(dim_z, n_win, DATA_FILE)

    device = torch.device("cuda:0")
    model = TestAutoencoder(config=config, encoder=encoder, decoder=decoder)

    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_ae")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=25,
        logger=tensorboard
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()