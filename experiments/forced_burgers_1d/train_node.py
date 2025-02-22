import pytorch_lightning as pl
import torch
import math
from torchdiffeq import odeint

from dataclasses import dataclass
from torch import Tensor, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from jaxtyping import jaxtyped
from beartype import beartype

import os
import pickle as pkl
import pathlib

from experiments.forced_burgers_1d.train_autoenc import TestAutoencoder, create_autoencoder
from experiments.forced_burgers_1d.def_model import DriftNet

import visde
# ruff: noqa: F821, F722

torch.manual_seed(42)
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data_ll_200_40_50.pkl"

@dataclass(frozen=True)
class NODEConfig:
    lr: float
    lr_sched_freq: int

class ODE(nn.Module):
    def __init__(self, drift, test_mu, test_t, test_f):
        super().__init__()
        self.drift = drift
        self.test_mu = test_mu
        self.test_t = test_t
        self.test_f = test_f
    
    def _forcing(self, t):
        return visde.utils.linterp(self.test_f.squeeze(0), self.test_t.squeeze(), t.squeeze(1))

    # Drift
    def forward(self, t, y):
        t = t.repeat(y.shape[0], 1)
        mu = self.test_mu.repeat(y.shape[0], 1)
        fi = self._forcing(t)
        return self.drift(mu, t, y, fi)  # shape (n_batch, dim_z)

class TestNODE(pl.LightningModule):
    """Variational inference of latent SDE"""

    _empty_tensor: Tensor  # empty tensor to get device
    _iter_count: int  # iteration count
    config: NODEConfig
    encoder: visde.VarEncoder
    decoder: visde.VarDecoder
    drift: visde.LatentDrift

    def __init__(
        self,
        config: NODEConfig,
        encoder: visde.VarEncoder,
        decoder: visde.VarDecoder,
        drift: visde.LatentDrift,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.drift = drift

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

        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        # t.shape can either be (n_batch,) or (n_batch, 1); this ensures the latter
        #if t.ndim == 1:
        #    t = t.unsqueeze(-1)

        temp_ode = ODE(self.drift, mu[0:1], t, f)

        with torch.no_grad():
            z_enc_mean, _ = self.encoder(mu, x_win)
        z_int = odeint(temp_ode, z_enc_mean[0:1], t).squeeze(1)
        raw_mse = (z_int - z_enc_mean).pow(2).mean()

        self.log("train/mse", raw_mse.item(), prog_bar=True)

        return raw_mse
    
    def validation_step(self,
                        batch: list[Tensor],
                        batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single validation step"""

        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        # t.shape can either be (n_batch,) or (n_batch, 1); this ensures the latter
        #if t.ndim == 1:
        #    t = t.unsqueeze(-1)
        
        temp_ode = ODE(self.drift, mu[0:1], t, f)

        with torch.no_grad():
            z_enc_mean, _ = self.encoder(mu, x_win)
        z_int = odeint(temp_ode, z_enc_mean[0:1], t).squeeze(1)
        raw_mse = (z_int - z_enc_mean).pow(2).mean()

        self.log("val/mse", raw_mse.item(), prog_bar=True)

        return raw_mse

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

def create_node(dim_z: int, data_file: str) -> tuple[visde.LatentDrift, visde.LatentDispersion]:
    with open(os.path.join(CURR_DIR, data_file), "rb") as f:
        data = pkl.load(f)
    
    dim_mu = data["train_mu"].shape[-1]
    dim_f = data["train_f"].shape[-1]

    driftconfig = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    driftnet = DriftNet(driftconfig)
    drift = visde.LatentDriftNoPrior(driftconfig, driftnet)

    return drift

def main():
    dim_z = 9
    n_win = 1
    n_batch = 128
    device = torch.device("cuda:0")

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch)

    # train autoencoder
    aeconfig, encoder, decoder = create_autoencoder(dim_z, n_win, DATA_FILE)
    ae_model = TestAutoencoder(config=aeconfig,
                               encoder=encoder,
                               decoder=decoder).to(device)

    ae_trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=50,
        logger=loggers.TensorBoardLogger(CURR_DIR, name="logs_node_ae"),
    )
    ae_trainer.fit(ae_model, train_dataloader, val_dataloader)

    # train node
    nodeconfig = NODEConfig(lr=1e-3, lr_sched_freq=5000)

    drift = create_node(dim_z, DATA_FILE)
    node_model = TestNODE(config=nodeconfig,
                          encoder=ae_model.encoder,
                          decoder=ae_model.decoder,
                          drift=drift).to(device)
    
    node_trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=50,
        logger=loggers.TensorBoardLogger(CURR_DIR, name="logs_node"),
        check_val_every_n_epoch=10
    )
    node_trainer.fit(node_model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()