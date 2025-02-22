import torch
from torch import Tensor

import visde
from tests.test_data import create_exp_dataloader
import pytorch_lightning as pl
from pytorch_lightning import loggers

import torchsde
from matplotlib import pyplot as plt

from tests.test_autoencoder import EncodeMeanNet, EncodeVarNet, DecodeMeanNet, DecodeVarNet
from tests.test_sdevar import LatentVarMeanNet, LatentVarVarNet, KernelNet
from tests.test_sdeprior import DriftNet, DispNet


def create_latent_sde(dim_mu, dim_x, dim_z, dim_f, n_batch, n_win, latentvar_type):
    # encoder
    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu, dim_x=dim_x, dim_z=dim_z, shape_x=(dim_x,), n_win=n_win)
    encode_mean_net = EncodeMeanNet(vaeconfig)
    encode_var_net = EncodeVarNet(vaeconfig)
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)

    # decoder
    decode_mean_net = DecodeMeanNet(vaeconfig)
    decode_var_net = DecodeVarNet(vaeconfig)
    decoder = visde.VarDecoderNoPrior(vaeconfig, decode_mean_net, decode_var_net)

    # drift
    config = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    driftnet = DriftNet(config)
    drift = visde.LatentDriftNoPrior(config, driftnet)

    # dispersion
    config = visde.LatentDispersionConfig(dim_mu=dim_mu, dim_z=dim_z)
    dispnet = DispNet(config)
    dispersion = visde.LatentDispersionNoPrior(config, dispnet)

    # likelihood
    loglikelihood = visde.LogLikeGaussian()

    # latent variational distribution
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    if latentvar_type == 'latentgp':
        z_mean_net = LatentVarMeanNet(config)
        z_var_net = LatentVarVarNet(config)
        latentvar = visde.LatentVarGP(config, z_mean_net, z_var_net)
    elif latentvar_type == 'amortized':
        kernel_net = KernelNet(config)
        kernel = visde.DeepGaussianKernel(kernel_net, n_batch, 0.1)
        latentvar = visde.AmortizedLatentVarGP(config, kernel, encoder)
    elif latentvar_type == 'paramfree':
        latentvar = visde.ParamFreeLatentVarGP(config, encoder)

    config = visde.LatentSDEConfig(n_totaldata=5*251, n_samples=10, n_tquad=0, n_warmup=0, n_transition=10, lr=0.001, lr_sched_freq=100)
    model = visde.LatentSDE(config=config,
                            encoder=encoder,
                            decoder=decoder,
                            drift=drift,
                            dispersion=dispersion,
                            loglikelihood=loglikelihood,
                            latentvar=latentvar)
    
    return model

def test_training_latent_sde():
    dim_mu = 2
    dim_x = 1
    dim_z = 1
    dim_f = 1
    n_win = 3
    n_batch = 64
    latentvar_type = 'paramfree'
    
    dataloader = create_exp_dataloader(dim_mu, dim_x, n_win, n_batch)
    device = torch.device("cpu")
    model = create_latent_sde(dim_mu, dim_x, dim_z, dim_f, n_batch, n_win, latentvar_type)

    tensorboard = loggers.TensorBoardLogger("logs/")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=500,
        logger=tensorboard, #callbacks=[checkpoint_callback, summarizer_callback],
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, dataloader)

    t_size = 251

    mu_test = torch.ones(1, dim_mu)
    t_test = torch.linspace(0, 5, t_size).unsqueeze(0)
    f_test = torch.zeros(1, t_size, dim_f)
    sde = visde.SDE(model.drift, model.dispersion, mu_test, t_test, f_test)
    x0 = torch.full([1, n_win, dim_x], 1.0)
    z0 = model.encoder.sample(n_batch, mu_test, x0).squeeze(0)
    ts = torch.linspace(0, 1, t_size)
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    zs = torchsde.sdeint(sde, z0, ts)
    assert isinstance(zs, Tensor), "zs is expected to be a single tensor"
    mu_test = mu_test.unsqueeze(0).repeat(zs.shape[0], zs.shape[1], 1)
    xs = model.decoder.sample(1, mu_test.flatten(0, 1), zs.flatten(0, 1)).unflatten(0, (t_size, n_batch))
    print(xs.flatten(1).shape)

    fig, ax = plt.subplots()
    ax.plot(ts, xs.flatten(1).detach().numpy())
    plt.savefig("tests/test_sde.png")
    plt.show()

if __name__ == "__main__":
    test_training_latent_sde()