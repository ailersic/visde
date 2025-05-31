import pytorch_lightning as pl
import torch
import math
from torchdiffeq import odeint#_adjoint as odeint

from dataclasses import dataclass
from torch import Tensor, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
#from pytorch_lightning.callbacks import EarlyStopping
from jaxtyping import jaxtyped
from beartype import beartype

import os
import pickle as pkl
import pathlib
import numpy as np
import shutil

from datasets.forced_burgers_1d.load_data import load_data
from experiments.forced_burgers_1d.visde.def_model import DriftNetNODE, EncodeMeanNet, DecodeMeanNet, EncodeVarNet, DecodeVarNet

import visde
# ruff: noqa: F821, F722

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

@dataclass(frozen=True)
class PNODEConfig:
    lr: float
    lr_sched_freq: int

class PNODE(nn.Module):
    def __init__(self, drift, test_mu):
        super().__init__()
        self.drift = drift
        self.test_mu = test_mu
    
    # Drift
    def forward(self, t, z):
        t = t.repeat(z.shape[0], 1)
        mu = self.test_mu.repeat(z.shape[0], 1)
        f = torch.zeros_like(t).to(z.device) # dummy forcing term
        return self.drift(mu, t, z, f)  # shape (n_batch, dim_z)

class TestPNODE(pl.LightningModule):
    """Variational inference of latent SDE"""

    _empty_tensor: Tensor  # empty tensor to get device
    _iter_count: int  # iteration count
    config: PNODEConfig
    encoder: nn.Module
    decoder: nn.Module
    drift: nn.Module

    def __init__(
        self,
        config: PNODEConfig,
        encoder: visde.VarEncoderNoPrior,
        decoder: visde.VarDecoderNoPrior,
        drift: nn.Module,
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

    def gauss_loglikelihood(self, x: Tensor, mu: Tensor, var: Tensor) -> Tensor:
        """Calculate the log-likelihood of a Gaussian distribution"""
        return -0.5 * torch.log(2 * math.pi * var) - (x - mu).pow(2) / (2 * var)

    @jaxtyped(typechecker=beartype)
    def training_step(self,
                      batch: list[Tensor],
                      batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single training step"""

        self._iter_count += 1

        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        temp_ode = PNODE(self.drift, mu[0:1])

        z_enc_mean, z_enc_var = self.encoder(mu, x_win)
        z_enc_samp = self.encoder.sample(1, mu, x_win)
        z_enc_0 = z_enc_samp[0].unsqueeze(0)

        z_int = odeint(temp_ode, z_enc_0, t).squeeze(1)
        x_pred_mean, x_pred_var = self.decoder(mu, z_int)

        loglike_x = self.gauss_loglikelihood(x, x_pred_mean, x_pred_var)
        kl_z = 0.5 * (z_enc_mean.pow(2) + z_enc_var - torch.log(z_enc_var) - 1).sum(-1)

        elbo = loglike_x.mean() - kl_z.mean()

        self.log("train/loglike", loglike_x.mean(), prog_bar=True, logger=True)
        self.log("train/kl_div", kl_z.mean(), prog_bar=True, logger=True)
        self.log("train/elbo", elbo, prog_bar=True, logger=True)

        return -elbo  # negative ELBO for minimization
    
    def validation_step(self,
                        batch: list[Tensor],
                        batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single validation step"""

        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        temp_ode = PNODE(self.drift, mu[0:1])

        z_enc_mean, z_enc_var = self.encoder(mu, x_win)
        z_enc_samp = self.encoder.sample(1, mu, x_win)
        z_enc_0 = z_enc_samp[0].unsqueeze(0)

        z_int = odeint(temp_ode, z_enc_0, t).squeeze(1)
        x_pred_mean, x_pred_var = self.decoder(mu, z_int)

        loglike_x = self.gauss_loglikelihood(x, x_pred_mean, x_pred_var)
        kl_z = 0.5 * (z_enc_mean.pow(2) + z_enc_var - torch.log(z_enc_var) - 1).sum(-1)

        elbo = loglike_x.mean() - kl_z.mean()

        self.log("val/loglike", loglike_x.mean(), prog_bar=True, logger=True)
        self.log("val/kl_div", kl_z.mean(), prog_bar=True, logger=True)
        self.log("val/elbo", elbo, prog_bar=True, logger=True)

        return -elbo  # negative ELBO for minimization

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

def get_dataloaders(n_win: int, n_batch: int, data_file: str) -> tuple[DataLoader, DataLoader]:
    data = load_data(data_file)

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

def create_autoencoder(dim_z: int, n_win: int, data_file: str) -> tuple[nn.Module, nn.Module]:
    data = load_data(data_file)
    
    dim_mu = data["train_mu"].shape[1]
    shape_x = tuple(data["train_x"].shape[2:])
    dim_x = int(np.prod(shape_x))

    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu,
                                           dim_x=dim_x,
                                           dim_z=dim_z,
                                           shape_x=shape_x,
                                           n_win=n_win)

    #data_mean, data_std = torch.mean(data["train_x"]), torch.std(data["train_x"])

    encode_mean_net = EncodeMeanNet(vaeconfig)
    encode_var_net = EncodeVarNet(vaeconfig)
    with torch.no_grad():
        encode_var_net.fixed_var = torch.nn.Parameter(torch.zeros((1, dim_z)))
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)
    
    decode_mean_net = DecodeMeanNet(vaeconfig)
    decode_var_net = DecodeVarNet(vaeconfig)
    decoder = visde.VarDecoderNoPrior(vaeconfig, decode_mean_net, decode_var_net)

    return encoder, decoder

def create_pnode(dim_z: int, data_file: str) -> nn.Module:
    data = load_data(data_file)
    
    dim_mu = data["train_mu"].shape[-1]
    dim_f = data["train_f"].shape[-1]

    driftconfig = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    drift = DriftNetNODE(driftconfig)

    return drift

def main(overwrite: bool = False, data_file: str = "data_bcf_100_10_10_param.pkl", dim_z = 3):
    n_win = 1
    n_batch = 1001
    print("CUDA:", torch.cuda.is_available())

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch, data_file)

    version = "_".join([data_file.split(".")[0], str(dim_z), str(n_batch), "50"])
    '''
    if os.path.exists(os.path.join(CURR_DIR, "logs_pnode", version)):
        if overwrite:
            print(f"Version {version} already exists. Overwriting...", flush=True)
            while os.path.exists(os.path.join(CURR_DIR, "logs_pnode", version)):
                try:
                    shutil.rmtree(os.path.join(CURR_DIR, "logs_pnode", version))
                except OSError as e:
                    continue
        else:
            print(f"Version {version} already exists. Skipping...", flush=True)
            return
    '''

    ckpt_dir = os.path.join(CURR_DIR, "logs_pnode", version, "checkpoints")
    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            ckpt_file = file

    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_pnode", version=version)

    # train node
    pnodeconfig = PNODEConfig(lr=1e-3, lr_sched_freq=2000)

    encoder, decoder = create_autoencoder(dim_z, n_win, data_file)
    drift = create_pnode(dim_z, data_file)
    pnode_model = TestPNODE(config=pnodeconfig,
                            encoder=encoder,
                            decoder=decoder,
                            drift=drift).to(device)
    
    pnode_trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=50,
        logger=tensorboard,
        check_val_every_n_epoch=5,
        #callbacks=[EarlyStopping(monitor="val/mse", mode="min")]
    )
    pnode_trainer.fit(pnode_model, train_dataloader, val_dataloader, ckpt_path=os.path.join(ckpt_dir, ckpt_file))

if __name__ == "__main__":
    main()