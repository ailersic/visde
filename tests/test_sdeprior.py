import torch
from torch import Tensor, nn

import visde

class DriftNet(nn.Module):
    def __init__(self, config):
        super(DriftNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z
        dim_f = config.dim_f

        self.net = nn.Sequential(nn.Linear(dim_mu + dim_z + dim_f, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_z))

    def forward(self, mu: Tensor, t: Tensor, z: Tensor, f: Tensor) -> Tensor:
        h = torch.cat([mu, z, f], dim=-1)
        dz = self.net(h)
        return dz

class DispNet(nn.Module):
    def __init__(self, config):
        super(DispNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z

        self.net = nn.Sequential(nn.Linear(dim_mu + 1, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_z))

    def forward(self, mu: Tensor, t: Tensor) -> Tensor:
        h = torch.cat([mu, t], dim=-1)
        dz = self.net(h)
        return torch.exp(dz)

def test_latentdrift():
    dim_mu = 5
    dim_z = 5
    dim_f = 1
    config = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    driftnet = DriftNet(config)
    drift = visde.LatentDriftNoPrior(config, driftnet)

    assert drift.device == torch.device("cpu")

    batch_size = 2
    mu = torch.randn(batch_size, dim_mu)
    t = torch.rand(batch_size, 1)
    z = torch.randn(batch_size, dim_z)
    f = torch.randn(batch_size, dim_f)
    assert drift(mu, t, z, f).shape == (batch_size, dim_z)
    assert drift.kl_divergence().shape == ()
    print(drift(mu, t, z, f))

def test_latentdispersion():
    dim_mu = 5
    dim_z = 5
    config = visde.LatentDispersionConfig(dim_mu=dim_mu, dim_z=dim_z)
    dispnet = DispNet(config)
    dispersion = visde.LatentDispersionNoPrior(config, dispnet)

    assert dispersion.device == torch.device("cpu")

    batch_size = 2
    mu = torch.randn(batch_size, dim_mu)
    t = torch.rand(batch_size, 1)
    assert dispersion(mu, t).shape == (batch_size, dim_z)
    assert dispersion.kl_divergence().shape == ()
    print(dispersion(mu, t))
    
if __name__ == "__main__":
    test_latentdrift()
    test_latentdispersion()