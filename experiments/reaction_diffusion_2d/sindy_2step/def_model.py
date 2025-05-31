import matplotlib.pyplot as plt
import pysindy as ps
import timeit
import os
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import visde
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from jaxtyping import jaxtyped
from beartype import beartype
import math

from experiments.reaction_diffusion_2d.visde.def_model import EncodeMeanNet, EncodeVarNet, DecodeMeanNet, DecodeVarNet
from datasets.reaction_diffusion_2d.load_data import load_data

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
        raw_mse = (x_rec_mean - x.to(self.device)).pow(2).mean()
        norm_mse = raw_mse / (x.to(self.device).pow(2).mean())

        self.log("train/raw_rmse", raw_mse.sqrt().item(), prog_bar=True)
        self.log("train/norm_rmse", norm_mse.sqrt().item(), prog_bar=True)

        return raw_mse
    
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
        raw_mse = (x_rec_mean - x.to(self.device)).pow(2).mean()
        norm_mse = raw_mse / (x.to(self.device).pow(2).mean())

        self.log("val/raw_rmse", raw_mse.sqrt().item(), prog_bar=True)
        self.log("val/norm_rmse", norm_mse.sqrt().item(), prog_bar=True)

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

def create_autoencoder(dim_z: int, n_win: int, data_file: str) -> tuple[TestAutoencoderConfig, visde.VarEncoder, visde.VarDecoder]:
    data = load_data(data_file)
    
    dim_mu = data["train_mu"].shape[1]
    shape_x = tuple(data["train_x"].shape[2:])
    dim_x = int(np.prod(shape_x))

    config = TestAutoencoderConfig(n_totaldata=torch.numel(data["train_t"]),
                                   n_samples=1,
                                   n_tquad=0,
                                   n_warmup=0,
                                   n_transition=9000,
                                   lr=1e-3,
                                   lr_sched_freq=2000)

    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu,
                                           dim_x=dim_x,
                                           dim_z=dim_z,
                                           shape_x=shape_x,
                                           n_win=n_win)

    encode_mean_net = EncodeMeanNet(vaeconfig)
    encode_var_net = EncodeVarNet(vaeconfig)
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)
    
    decode_mean_net = DecodeMeanNet(vaeconfig)
    decode_var_net = DecodeVarNet(vaeconfig)
    decoder = visde.VarDecoderNoPrior(vaeconfig, decode_mean_net, decode_var_net)

    return config, encoder, decoder

def create_sindy_model(x, dx, u, dt, threshold, degree):
    dim_x = x[0].shape[1]
    dim_u = u[0].shape[1]

    feature_lib = ps.PolynomialLibrary(degree=degree, include_bias=False)
    '''
    parameter_lib = ps.PolynomialLibrary(degree=1, include_interaction=False, include_bias=False)

    lib = ps.ParameterizedLibrary(
        feature_library=feature_lib,
        parameter_library=parameter_lib,
        num_features=dim_x,
        num_parameters=dim_u,
    )
    '''
    opt = ps.STLSQ(threshold=threshold, max_iter=1000)
    model = ps.SINDy(
        feature_library=feature_lib,
        optimizer=opt,
        feature_names=[f"x{i}" for i in range(dim_x)],
        discrete_time=False,
    )

    model.fit(x, t=dt, x_dot=dx, multiple_trajectories=True)

    return model
