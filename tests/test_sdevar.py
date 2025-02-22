import torch
from torch import Tensor, nn

import visde
from tests.test_autoencoder import EncodeMeanNet, EncodeVarNet

class LatentVarMeanNet(nn.Module):
    def __init__(self, config):
        super(LatentVarMeanNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z

        self.net = nn.Sequential(nn.Linear(dim_mu + 1, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_z))

    def forward(self, mu: Tensor, t: Tensor) -> Tensor:
        mu_t = torch.cat([mu, t], dim=-1)
        z = self.net(mu_t)
        return z

class LatentVarVarNet(nn.Module):
    def __init__(self, config):
        super(LatentVarVarNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z

        self.net = nn.Sequential(nn.Linear(dim_mu + 1, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_z))

    def forward(self, mu: Tensor, t: Tensor) -> Tensor:
        mu_t = torch.cat([mu, t], dim=-1)
        z = self.net(mu_t)
        return torch.exp(z)

class KernelNet(nn.Module):
    def __init__(self, config):
        super(KernelNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(1, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, t: Tensor) -> Tensor:
        tk = self.net(t)
        return tk

def create_varencoder(dim_mu, dim_x, dim_z, n_win):
    config = visde.VarAutoencoderConfig(dim_mu, dim_x=dim_x, dim_z=dim_z, shape_x=(dim_x,), n_win=n_win)
    encode_mean_net = EncodeMeanNet(config)
    encode_var_net = EncodeVarNet(config)
    encoder = visde.VarEncoderNoPrior(config, encode_mean_net, encode_var_net)
    
    return encoder

def test_amortizedlatentvargp():
    dim_mu = 5
    dim_x = 1
    dim_z = 2
    n_win = 3
    n_batch = 4
    dt = 0.1
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    kernel_net = KernelNet(config)
    kernel = visde.DeepGaussianKernel(kernel_net, n_batch, dt)
    encoder = create_varencoder(dim_mu=dim_mu, dim_x=dim_x, dim_z=dim_z, n_win=n_win)
    latentvargp = visde.AmortizedLatentVarGP(config, kernel, encoder)
    
    assert latentvargp.device == torch.device("cpu")
    
    x_win = torch.randn(n_batch, n_win, dim_x)
    for i in range(1, n_batch):
        x_win[i, 0:n_win-1] = x_win[i-1, 1:n_win]
    mu = torch.randn(n_batch, dim_mu)
    t = torch.linspace(0, 1, n_batch).unsqueeze(-1)
    latentvargp.form_window(mu, t, x_win)
    print("Form window passed")
    
    t = torch.rand(n_batch).unsqueeze(-1)
    z_mean, z_var, z_dmean, z_dvar = latentvargp(mu, t)
    print(z_mean.shape)
    print(z_var.shape)
    print(z_dmean.shape)
    print(z_dvar.shape)
    assert z_mean.shape == (n_batch, dim_z)
    assert z_var.shape == (n_batch, dim_z)
    assert z_dmean.shape == (n_batch, dim_z)
    assert z_dvar.shape == (n_batch, dim_z)
    print("Forward passed")

    td = torch.linspace(0, 0.0001*(n_batch-1), n_batch).unsqueeze(-1)
    z_mean, z_var, z_dmean, z_dvar = latentvargp(mu, td)
    print((z_mean[2] - z_mean[0])/0.0002)
    print(z_dmean[1])
    print("Derivative check")

    n_samples = 10
    z_samples = latentvargp.sample(n_samples, mu, t)
    print(z_samples.shape)
    assert z_samples.shape == (n_batch*n_samples, dim_z)
    print("Sample passed")

def test_latentvargp():
    dim_mu = 5
    dim_z = 5
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    z_mean_net = LatentVarMeanNet(config)
    z_var_net = LatentVarVarNet(config)
    latentvar = visde.LatentVarGP(config, z_mean_net, z_var_net)
    
    assert latentvar.device == torch.device("cpu")
    
    n_batch = 2
    t = torch.rand(n_batch, 1)
    mu = torch.randn(n_batch, dim_mu)
    z_mean, z_var, z_dmean, z_dvar = latentvar(mu, t)
    print(z_mean.shape)
    print(z_var.shape)
    print(z_dmean.shape)
    print(z_dvar.shape)
    assert z_mean.shape == (n_batch, dim_z)
    assert z_var.shape == (n_batch, dim_z)
    assert z_dmean.shape == (n_batch, dim_z)
    assert z_dvar.shape == (n_batch, dim_z)
    print("Forward passed")

    n_samples = 10
    z_samples = latentvar.sample(n_samples, mu, t)
    print(z_samples.shape)
    assert z_samples.shape == (n_batch*n_samples, dim_z)
    print("Sample passed")
    
if __name__ == "__main__":
    print("---LATENTVARGP---")
    test_latentvargp()
    print("---AMORTIZEDLATENTVARGP---")
    test_amortizedlatentvargp()