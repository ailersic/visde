import torch
from torch import Tensor, nn
from jaxtyping import Float

import numpy as np
import pickle as pkl
import os
import pathlib

import visde
# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

class EncodeMeanNet(nn.Module):
    def __init__(self, config, data_mean, data_std):
        super(EncodeMeanNet, self).__init__()
        dim_z = config.dim_z
        n_win = config.n_win
        shape_x = config.shape_x
        chan_x = shape_x[0]

        offset = data_mean*torch.ones((1, n_win*chan_x, *shape_x[1:]))
        scale = data_std*torch.ones((1, n_win*chan_x, *shape_x[1:]))
        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)

        self.convnet = nn.Sequential(nn.Conv2d(n_win*chan_x, 32, 5, stride=2, padding=2),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 5, stride=2, padding=2),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 128, 5, stride=2, padding=2),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 256, 5, stride=2, padding=2),
                                     nn.ReLU(),
                                     nn.Conv2d(256, 512, 5, stride=2, padding=2),
                                     nn.ReLU(),
                                     nn.Flatten())
        
        dim_fc_in = 3*14*512

        self.fcnet = nn.Sequential(nn.Linear(dim_fc_in, dim_z))
        
        for layer in self.convnet:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.fcnet:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        x_win_reshape = x_win.view(x_win.shape[0], -1, x_win.shape[-2], x_win.shape[-1])
        x_win_norm = (x_win_reshape - self.offset) / self.scale
        h = self.convnet(x_win_norm)
        z = self.fcnet(h)
        return z

class EncodeVarNet(nn.Module):
    def __init__(self, config, data_mean, data_std):
        super(EncodeVarNet, self).__init__()
        dim_z = config.dim_z
        n_win = config.n_win
        shape_x = config.shape_x
        chan_x = shape_x[0]

        offset = data_mean*torch.ones((1, n_win*chan_x, *shape_x[1:]))
        scale = data_std*torch.ones((1, n_win*chan_x, *shape_x[1:]))
        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)

        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4.0*torch.ones((1, dim_z)))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        z_var_norm = self.fixed_var.expand(x_win.shape[0], *self.fixed_var.shape[1:])
        z_var = self.out_activ(z_var_norm)
        return z_var

class DecodeMeanNet(nn.Module):
    def __init__(self, config, data_mean, data_std):
        super(DecodeMeanNet, self).__init__()
        dim_z = config.dim_z
        shape_x = config.shape_x
        dim_fc_in = 3*14*512
        chan_x = shape_x[0]

        offset = data_mean*torch.ones((1, *shape_x))
        scale = data_std*torch.ones((1, *shape_x))
        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)

        self.fcnet = nn.Sequential(nn.Linear(dim_z, dim_fc_in),
                                   nn.ReLU())
        
        self.convnet = nn.Sequential(nn.Unflatten(-1, (512, 3, 14)),
                                     nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=(0, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=(1, 0)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=(1, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=(1, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, chan_x, 5, stride=2, padding=2, output_padding=(1, 1)))
        
        for layer in self.convnet:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        for layer in self.fcnet:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, mu: Tensor, z: Tensor) -> Tensor:
        h = self.fcnet(z)
        x_mean = self.convnet(h) * self.scale + self.offset
        return x_mean

class DriftNet(nn.Module):
    def __init__(self, config):
        super(DriftNet, self).__init__()
        dim_z = config.dim_z
        dim_f = config.dim_f

        self.net = nn.Sequential(nn.Linear(dim_z + dim_f + 2, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, dim_z))

    def forward(self, mu: Tensor, t: Tensor, z: Tensor, f: Tensor) -> Tensor:
        h = torch.cat([z, f, torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t)], dim=-1)
        dz = self.net(h)
        return dz

class DispNet(nn.Module): # for NSDE only
    def __init__(self, config):
        super(DispNet, self).__init__()
        dim_z = config.dim_z

        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-6.0*torch.ones((1, dim_z)))

    def forward(self, mu: Tensor, t: Tensor) -> Tensor:
        dz = self.fixed_var.expand(t.shape[0], *self.fixed_var.shape[1:])
        return dz

class LatentDispersionWithLogNormalPrior(nn.Module): # for VISDE only
    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(
        self,
        config: visde.LatentDispersionConfig,
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.dim_mu = self.config.dim_mu
        self.dim_z = self.config.dim_z

        self.logdisp_prior_mean = torch.full((1, self.dim_z), -6.0).to(self.device)
        self.logdisp_prior_logvar = torch.full((1, self.dim_z), 0.0).to(self.device)
        self.logdisp_mean = nn.Parameter(torch.clone(self.logdisp_prior_mean).to(self.device))
        self.logdisp_logvar = nn.Parameter(torch.clone(self.logdisp_prior_logvar).to(self.device))
        self.resample_params()
        
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device
        
    def resample_params(self) -> None:
        stdnorm_samples = torch.randn_like(self.logdisp_mean, device=self.device)
        logdisp = self.logdisp_mean + torch.mul(stdnorm_samples, torch.exp(0.5*self.logdisp_logvar))
        self.disp = torch.exp(logdisp)
    
    def kl_divergence(self) -> Tensor:
        prior_mean = self.logdisp_prior_mean.to(self.device)
        prior_var = torch.exp(self.logdisp_prior_logvar).to(self.device)
        mean = self.logdisp_mean
        var = torch.exp(self.logdisp_logvar)

        kl_div = 0.5 * (prior_var.log().sum() - var.log().sum()
                        - var.numel()
                        + (var / prior_var).sum()
                        + ((mean - prior_mean).pow(2) / prior_var).sum())
        return kl_div
    
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                z: Float[Tensor, "n_batch dim_z"]
    ) -> tuple[Float[Tensor, "n_batch *shape_x"],
               Float[Tensor, "n_batch *shape_x"]
    ]:
        dz = self.disp.expand(z.shape[0], *self.disp.shape[1:])
        return dz

class KernelNet(nn.Module):
    def __init__(self, config):
        super(KernelNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(1, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 1))

    def forward(self, t: Tensor) -> Tensor:
        return self.net(t)

class VarDecoderWithLogNormalPrior(nn.Module):
    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(
        self,
        config: visde.VarAutoencoderConfig,
        decode_mean_net: nn.Module,
        data_mean: Tensor,
        data_std: Tensor,
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.dim_mu = self.config.dim_mu
        self.shape_x = self.config.shape_x
        self.dim_z = self.config.dim_z
        self.n_win = self.config.n_win

        self.decode_mean = decode_mean_net

        self.offset = data_mean*torch.ones((1, *self.shape_x)).to(self.device)
        self.scale = data_std*torch.ones((1, *self.shape_x)).to(self.device)

        self.dec_logvar_prior_mean = torch.full(self.shape_x, -5.0).to(self.device)
        self.dec_logvar_prior_logvar = torch.full(self.shape_x, 1.0).to(self.device)
        self.dec_logvar_mean = nn.Parameter(torch.clone(self.dec_logvar_prior_mean).to(self.device))
        self.dec_logvar_logvar = nn.Parameter(torch.clone(self.dec_logvar_prior_logvar).to(self.device))
        self.resample_params()
        
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device
        
    def resample_params(self) -> None:
        stdnorm_samples = torch.randn_like(self.dec_logvar_mean, device=self.device)
        dec_logvar = self.dec_logvar_mean + torch.mul(stdnorm_samples, torch.exp(0.5*self.dec_logvar_logvar))
        self.dec_var = torch.exp(dec_logvar)
    
    def kl_divergence(self) -> Tensor:
        prior_mean = self.dec_logvar_prior_mean.to(self.device)
        prior_var = torch.exp(self.dec_logvar_prior_logvar).to(self.device)
        mean = self.dec_logvar_mean
        var = torch.exp(self.dec_logvar_logvar)

        kl_div = 0.5 * (prior_var.log().sum() - var.log().sum()
                        - var.numel()
                        + (var / prior_var).sum()
                        + ((mean - prior_mean).pow(2) / prior_var).sum())
        return kl_div
    
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                z: Float[Tensor, "n_batch dim_z"]
    ) -> tuple[Float[Tensor, "n_batch *shape_x"],
               Float[Tensor, "n_batch *shape_x"]
    ]:
        x_mean = self.decode_mean(mu, z)
        x_var = self.dec_var.to(self.device).unsqueeze(0).expand(z.shape[0], *self.dec_var.shape).mul(self.scale.pow(2).to(self.device))

        return x_mean, x_var

    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               z: Float[Tensor, "n_batch dim_z"]
    ) -> Float[Tensor, "..."]:
        x_mean, x_var = self.forward(mu, z)
        n_x_dims = len(self.shape_x)

        x_mean = x_mean.unsqueeze(-n_x_dims-1)
        x_stdev = x_var.sqrt().unsqueeze(-n_x_dims-1).to(self.device)

        n_batch = x_mean.shape[0]
        stdnorm_samples = torch.randn(n_batch, n_samples, *self.shape_x, device=self.device)
        x = (x_mean + torch.mul(x_stdev, stdnorm_samples)).flatten(0, 1)
        
        # return samples with shape (n_batch*n_samples, *shape_x)
        return x

def create_latent_sde(dim_z: int,
                      n_batch: int,
                      n_win: int,
                      device: torch.device = torch.device("cuda:0")
) -> visde.sde.LatentSDE:
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    
    dim_mu = data["train_mu"].shape[1]
    shape_x = tuple(data["train_x"].shape[2:])
    dim_x = int(np.prod(shape_x))
    dim_f = data["train_f"].shape[2]
    dt = data["train_t"][0,1] - data["train_t"][0,0]

    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu,
                                           dim_x=dim_x,
                                           dim_z=dim_z,
                                           shape_x=shape_x,
                                           n_win=n_win)

    data_mean, data_std = torch.mean(data["train_x"]), torch.std(data["train_x"])

    # encoder
    encode_mean_net = EncodeMeanNet(vaeconfig, data_mean, data_std)
    encode_var_net = EncodeVarNet(vaeconfig, data_mean, data_std)
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)

    # decoder        
    decode_mean_net = DecodeMeanNet(vaeconfig, data_mean, data_std)
    decoder = VarDecoderWithLogNormalPrior(vaeconfig, decode_mean_net, data_mean, data_std)

    # drift
    driftconfig = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    driftnet = DriftNet(driftconfig)
    drift = visde.LatentDriftNoPrior(driftconfig, driftnet)

    # dispersion
    dispconfig = visde.LatentDispersionConfig(dim_mu=dim_mu, dim_z=dim_z)
    dispersion = LatentDispersionWithLogNormalPrior(dispconfig)

    # likelihood
    loglikelihood = visde.LogLikeGaussian()

    # latent variational distribution
    kernelconfig = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    kernelnet = KernelNet(kernelconfig)
    kernel = visde.DeepGaussianKernel(kernelnet, n_batch, dt)
    latentvar = visde.AmortizedLatentVarGP(kernelconfig, kernel, encoder)

    config = visde.LatentSDEConfig(n_totaldata=torch.numel(data["train_t"]),
                                   n_samples=1,
                                   n_tquad=10,
                                   n_warmup=0,
                                   n_transition=60000,
                                   lr=1e-3,
                                   lr_sched_freq=5000)
    model = visde.LatentSDE(config=config,
                            encoder=encoder,
                            decoder=decoder,
                            drift=drift,
                            dispersion=dispersion,
                            loglikelihood=loglikelihood,
                            latentvar=latentvar).to(device)
    
    return model
