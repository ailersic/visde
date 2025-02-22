import torch
from torch import Tensor, nn
import numpy as np

import visde

class EncodeMeanNet(nn.Module):
    def __init__(self, config):
        super(EncodeMeanNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z
        n_win = config.n_win
        shape_x = config.shape_x
        dim_x = np.prod(shape_x)

        self.net = nn.Sequential(nn.Linear(dim_x*n_win + dim_mu, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_z))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        x_mu = torch.cat([x_win.flatten(-2), mu], dim=-1)
        z = self.net(x_mu)
        return z

class EncodeVarNet(nn.Module):
    def __init__(self, config):
        super(EncodeVarNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z
        n_win = config.n_win
        shape_x = config.shape_x
        dim_x = np.prod(shape_x)

        self.net = nn.Sequential(nn.Linear(dim_x*n_win + dim_mu, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_z))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        x_mu = torch.cat([x_win.flatten(-2), mu], dim=-1)
        z = self.net(x_mu)
        return torch.exp(z)

class DecodeMeanNet(nn.Module):
    def __init__(self, config):
        super(DecodeMeanNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z
        dim_x = np.prod(config.shape_x)

        self.net = nn.Sequential(nn.Linear(dim_z + dim_mu, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_x))

    def forward(self, mu: Tensor, z: Tensor) -> Tensor:
        z_mu = torch.cat([z, mu], dim=-1)
        x = self.net(z_mu)
        return x

class DecodeVarNet(nn.Module):
    def __init__(self, config):
        super(DecodeVarNet, self).__init__()
        dim_mu = config.dim_mu
        dim_z = config.dim_z
        dim_x = np.prod(config.shape_x)

        self.net = nn.Sequential(nn.Linear(dim_z + dim_mu, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, dim_x))

    def forward(self, mu: Tensor, z: Tensor) -> Tensor:
        z_mu = torch.cat([z, mu], dim=-1)
        x = self.net(z_mu)
        return torch.exp(x)

def test_varencoder(config):
    dim_mu = config.dim_mu
    dim_x = config.dim_x
    dim_z = config.dim_z
    n_win = config.n_win

    encode_mean_net = EncodeMeanNet(config)
    encode_var_net = EncodeVarNet(config)
    encoder = visde.VarEncoderNoPrior(config, encode_mean_net, encode_var_net)

    assert encoder.device == torch.device("cpu")

    batch_size = 2
    x = torch.randn(batch_size, n_win, dim_x)
    mu = torch.randn(batch_size, dim_mu)
    assert encoder(mu, x)[0].shape == (batch_size, dim_z)
    assert encoder(mu, x)[1].shape == (batch_size, dim_z)
    print(encoder(mu, x)[0].shape, encoder(mu, x)[1].shape)
    print(encoder.sample(100, mu, x).shape)
    
def test_vardecoder(config):
    dim_mu = config.dim_mu
    dim_x = config.dim_x
    dim_z = config.dim_z

    decode_mean_net = DecodeMeanNet(config)
    decode_var_net = DecodeVarNet(config)
    decoder = visde.VarDecoderNoPrior(config, decode_mean_net, decode_var_net)

    assert decoder.device == torch.device("cpu")

    batch_size = 2
    z = torch.randn(batch_size, dim_z)
    mu = torch.randn(batch_size, dim_mu)
    assert decoder(mu, z)[0].shape == (batch_size, dim_x)
    assert decoder(mu, z)[1].shape == (batch_size, dim_x)
    print(decoder(mu, z)[0].shape, decoder(mu, z)[1].shape)
    print(decoder.sample(100, mu, z).shape)

def main():
    dim_mu = 4
    dim_x = 10
    dim_z = 5
    n_win = 3
    config = visde.VarAutoencoderConfig(dim_mu=dim_mu, dim_x=dim_x, dim_z=dim_z, n_win=n_win, shape_x=(dim_x,))

    test_varencoder(config)
    test_vardecoder(config)

if __name__ == "__main__":
    main()