import torch
from torch import Tensor, nn

import visde

class KernelNet(nn.Module):
    def __init__(self, config):
        super(KernelNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(1, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, t: Tensor) -> Tensor:
        tk = self.net(t)
        return tk

def test_kernel():
    dim_mu = 5
    #dim_x = 10
    dim_z = 5
    n_batch = 2
    n_win = 3
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    kernel_net = KernelNet(config)
    
    #assert kernel_net.device == torch.device("cpu")
    
    #x_win = torch.randn(n_batch, n_win, dim_x)
    #mu = torch.randn(n_batch, dim_mu)
    t1, _ = torch.sort(torch.rand(n_batch, 1), dim=0)
    t2, _ = torch.sort(torch.rand(n_win, 1), dim=0)
    kernel = visde.DeepGaussianKernel(kernel_net, n_batch, 0.1)
    k = kernel(t1, t2)
    assert k.shape == (n_batch, n_win)
    print("Forward passed")
    
if __name__ == "__main__":
    test_kernel()