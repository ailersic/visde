from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.patches import Ellipse
# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
plt.rcParams.update({'font.size': 14})

def dyn(mu: Float[Tensor, "n_traj dim_mu"],
      t: Float[Tensor, "n_traj"],
      x: Float[Tensor, "n_traj 2 dim_x dim_x"]
) -> Float[Tensor, "n_traj 2 dim_x dim_x"]:
    beta = mu[:, 0].reshape(-1, 1, 1)
    d1 = mu[:, 1].reshape(-1, 1, 1)
    d2 = mu[:, 2].reshape(-1, 1, 1)

    dim_x = x.shape[-1]
    h = 20.0/(dim_x - 1)
    dxdt = torch.zeros_like(x)

    u = x[:, 0]
    v = x[:, 1]

    uxx = (u[:, 2:dim_x, 1:dim_x-1] + u[:, 0:dim_x-2, 1:dim_x-1] - 2*u[:, 1:dim_x-1, 1:dim_x-1])/(h**2)
    uyy = (u[:, 1:dim_x-1, 2:dim_x] + u[:, 1:dim_x-1, 0:dim_x-2] - 2*u[:, 1:dim_x-1, 1:dim_x-1])/(h**2)

    vxx = (v[:, 2:dim_x, 1:dim_x-1] + v[:, 0:dim_x-2, 1:dim_x-1] - 2*v[:, 1:dim_x-1, 1:dim_x-1])/(h**2)
    vyy = (v[:, 1:dim_x-1, 2:dim_x] + v[:, 1:dim_x-1, 0:dim_x-2] - 2*v[:, 1:dim_x-1, 1:dim_x-1])/(h**2)

    dxdt[:, 0, 1:dim_x-1, 1:dim_x-1] = ((1 - (u**2 + v**2))*u + beta*(u**2 + v**2)*v)[:, 1:-1, 1:-1] + d1*(uxx + uyy)
    dxdt[:, 1, 1:dim_x-1, 1:dim_x-1] = ((1 - (u**2 + v**2))*v - beta*(u**2 + v**2)*u)[:, 1:-1, 1:-1] + d2*(vxx + vyy)

    return dxdt

def create_dataset(mu: Float[Tensor, "n_traj dim_mu"],
                   T: float,
                   n_tstep: int,
                   dim_x: int
) -> tuple[Float[Tensor, "n_traj n_tstep"],
           Float[Tensor, "n_traj n_tstep dim_x"],
           Float[Tensor, "n_traj n_tstep dim_f"]
]:
    n_traj = mu.shape[0]
    t = torch.linspace(0.0, T, n_tstep).unsqueeze(0).repeat(n_traj, 1)
    f = torch.zeros(n_traj, n_tstep, 1)

    x = torch.zeros(n_traj, n_tstep, 2, dim_x, dim_x)
    xlin = torch.linspace(-10.0, 10.0, dim_x).unsqueeze(0).unsqueeze(2).repeat(n_traj, 1, dim_x)
    ylin = torch.linspace(-10.0, 10.0, dim_x).unsqueeze(0).unsqueeze(1).repeat(n_traj, dim_x, 1)
    
    m = 1
    gauss = torch.exp(-0.1*(xlin**2 + ylin**2))
    x[:, 0, 0, :, :] = gauss * torch.tanh(torch.sqrt(xlin**2 + ylin**2))*torch.cos(m*torch.atan2(xlin, ylin) - (torch.sqrt(xlin**2 + ylin**2)))
    x[:, 0, 1, :, :] = gauss * torch.tanh(torch.sqrt(xlin**2 + ylin**2))*torch.cos(m*torch.atan2(xlin, ylin) - (torch.sqrt(xlin**2 + ylin**2)))

    for i in range(1, n_tstep):
        dt = (t[:, i] - t[:, i - 1]).reshape(-1, 1, 1, 1)
        x[:, i] = x[:, i - 1] + dyn(mu, t[:, i - 1], x[:, i - 1]) * dt
        x[:, i] = x[:, i]
    
    x = gauss.unsqueeze(1).unsqueeze(2) * x
    x += 1e-2*torch.randn_like(x)
    
    return t, x, f

def main():
    n_traj = 1
    n_tstep = 1001
    dim_x = 100

    train_T = 50.0
    train_mu = torch.cat([torch.linspace(1.0, 1.0, n_traj).unsqueeze(1),
                          torch.linspace(0.1, 0.1, n_traj).unsqueeze(1),
                          torch.linspace(0.1, 0.1, n_traj).unsqueeze(1)], dim=1)

    val_T = 50.0
    val_mu = torch.cat([torch.full([1, 1], 1.0),
                        torch.full([1, 1], 0.1),
                        torch.full([1, 1], 0.1)], dim=1)
    
    test_T = 50.0
    test_mu = torch.cat([torch.full([1, 1], 1.0),
                         torch.full([1, 1], 0.1),
                         torch.full([1, 1], 0.1)], dim=1)

    train_t, train_x, train_f = create_dataset(train_mu, train_T, n_tstep, dim_x)
    val_t, val_x, val_f = create_dataset(val_mu, val_T, n_tstep, dim_x)
    test_t, test_x, test_f = create_dataset(test_mu, test_T, n_tstep, dim_x)

    assert train_mu.shape == (n_traj, 3)
    assert train_t.shape == (n_traj, n_tstep)
    assert train_x.shape == (n_traj, n_tstep, 2, dim_x, dim_x)
    assert train_f.shape == (n_traj, n_tstep, 1)

    assert val_mu.shape == (1, 3)
    assert val_t.shape == (1, n_tstep)
    assert val_x.shape == (1, n_tstep, 2, dim_x, dim_x)
    assert val_f.shape == (1, n_tstep, 1)
    
    assert test_mu.shape == (1, 3)
    assert test_t.shape == (1, n_tstep)
    assert test_x.shape == (1, n_tstep, 2, dim_x, dim_x)
    assert test_f.shape == (1, n_tstep, 1)

    train_x = train_x[:, :, 0:1, :, :]
    val_x = val_x[:, :, 0:1, :, :]
    test_x = test_x[:, :, 0:1, :, :]

    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(train_x[0, 0, 0], cmap="hot")
    axs[0,1].imshow(train_x[0, n_tstep//2, 0], cmap="hot")
    im = axs[0,2].imshow(train_x[0, -1, 0], cmap="hot")

    axs[1,0].imshow(val_x[0, 0, 0], cmap="hot")
    axs[1,1].imshow(val_x[0, n_tstep//2, 0], cmap="hot")
    axs[1,2].imshow(val_x[0, -1, 0], cmap="hot")

    axs[2,0].imshow(test_x[-1, 0, 0], cmap="hot")
    axs[2,1].imshow(test_x[-1, n_tstep//2, 0], cmap="hot")
    axs[2,2].imshow(test_x[-1, -1, 0], cmap="hot")
    plt.colorbar(im)
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "data.png"))

    data = {"train_mu": train_mu,
            "train_t": train_t,
            "train_x": train_x,
            "train_f": train_f,
            "val_mu": val_mu,
            "val_t": val_t,
            "val_x": val_x,
            "val_f": val_f,
            "test_mu": test_mu,
            "test_t": test_t,
            "test_x": test_x,
            "test_f": test_f}

    with open(os.path.join(CURR_DIR, "data.pkl"), "wb") as f:
        pkl.dump(data, f)

def plot_data(data):
    fig, axs = plt.subplots(1, 5, figsize=(12, 3), layout="constrained")
    axs[0].imshow(data["train_x"][0, 0, 0], cmap="coolwarm")
    axs[0].axis('off')
    axs[0].set_title(r"t=0")

    axs[1].imshow(data["train_x"][0, 200, 0], cmap="coolwarm")
    axs[1].axis('off')
    axs[1].set_title(r"t=10")

    axs[2].imshow(torch.zeros_like(data["train_x"][0, 0, 0]), cmap="binary")
    axs[2].add_patch(Ellipse((80, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
    axs[2].add_patch(Ellipse((50, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
    axs[2].add_patch(Ellipse((20, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
    axs[2].axis('off')

    axs[3].imshow(data["test_x"][0, -201, 0], cmap="coolwarm")
    axs[3].axis('off')
    axs[3].set_title(r"t=490")

    im = axs[4].imshow(data["test_x"][0, -1, 0], cmap="coolwarm")
    axs[4].axis('off')
    axs[4].set_title(r"t=500")

    fig.colorbar(im, ax=axs[4], shrink=0.6)

    #plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "data.pdf"))

def get_rd_data(noisy: bool = True):
    data = sio.loadmat(os.path.join(CURR_DIR, 'reaction_diffusion.mat'))

    train_n_tstep = 8000
    val_n_tstep = 1000
    test_n_tstep = data['t'].size - train_n_tstep - val_n_tstep
    dim_x = data['x'].size

    t = torch.linspace(0, data['t'][-1, 0], data['t'].size).unsqueeze(0)
    x = torch.tensor(data['uf']).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    filename = "data"

    if noisy:
        x += 1e-2*torch.randn_like(x)
        filename = "data_noisy"

    train_mu = torch.full([1, 1], 1.0)
    val_mu = torch.full([1, 1], 1.0)
    test_mu = torch.full([1, 1], 1.0)

    train_t = t[:, :train_n_tstep]
    val_t = t[:, train_n_tstep:train_n_tstep+val_n_tstep]
    test_t = t[:, train_n_tstep+val_n_tstep:]

    train_x = x[:, :train_n_tstep]
    val_x = x[:, train_n_tstep:train_n_tstep+val_n_tstep]
    test_x = x[:, train_n_tstep+val_n_tstep:]

    train_f = torch.zeros_like(train_t).unsqueeze(-1)
    val_f = torch.zeros_like(val_t).unsqueeze(-1)
    test_f = torch.zeros_like(test_t).unsqueeze(-1)

    assert train_mu.shape == (1, 1)
    assert train_t.shape == (1, train_n_tstep)
    assert train_x.shape == (1, train_n_tstep, 1, dim_x, dim_x)
    assert train_f.shape == (1, train_n_tstep, 1)

    assert val_mu.shape == (1, 1)
    assert val_t.shape == (1, val_n_tstep)
    assert val_x.shape == (1, val_n_tstep, 1, dim_x, dim_x)
    assert val_f.shape == (1, val_n_tstep, 1)
    
    assert test_mu.shape == (1, 1)
    assert test_t.shape == (1, test_n_tstep)
    assert test_x.shape == (1, test_n_tstep, 1, dim_x, dim_x)
    assert test_f.shape == (1, test_n_tstep, 1)

    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(train_x[0, 0, 0], cmap="hot")
    axs[0,1].imshow(train_x[0, train_n_tstep//2, 0], cmap="hot")
    axs[0,2].imshow(train_x[0, -1, 0], cmap="hot")

    axs[1,0].imshow(val_x[0, 0, 0], cmap="hot")
    axs[1,1].imshow(val_x[0, val_n_tstep//2, 0], cmap="hot")
    axs[1,2].imshow(val_x[0, -1, 0], cmap="hot")

    axs[2,0].imshow(test_x[-1, 0, 0], cmap="hot")
    axs[2,1].imshow(test_x[-1, test_n_tstep//2, 0], cmap="hot")
    axs[2,2].imshow(test_x[-1, -1, 0], cmap="hot")

    plt.show()
    plt.savefig(os.path.join(CURR_DIR, filename + ".png"))

    data = {"train_mu": train_mu.to(torch.float32),
            "train_t": train_t.to(torch.float32),
            "train_x": train_x.to(torch.float32),
            "train_f": train_f.to(torch.float32),
            "val_mu": val_mu.to(torch.float32),
            "val_t": val_t.to(torch.float32),
            "val_x": val_x.to(torch.float32),
            "val_f": val_f.to(torch.float32),
            "test_mu": test_mu.to(torch.float32),
            "test_t": test_t.to(torch.float32),
            "test_x": test_x.to(torch.float32),
            "test_f": test_f.to(torch.float32)}

    plot_data(data)

    with open(os.path.join(CURR_DIR, filename + ".pkl"), "wb") as f:
        pkl.dump(data, f)

if __name__ == "__main__":
    #main()
    get_rd_data(False)