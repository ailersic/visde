from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
plt.rcParams.update({'font.size': 16})

def dyn(mu: Float[Tensor, "n_traj dim_mu"],
        t: Float[Tensor, "n_traj"],
        x: Float[Tensor, "n_traj dim_x"],
        f: Float[Tensor, "n_traj dim_f"]
) -> Float[Tensor, "n_traj dim_x"]:
    visc = mu[:, 0]
    sigma = mu[:, 1].unsqueeze(1)
    scale = mu[:, 2].unsqueeze(1)

    dim_x = x.shape[-1]
    h = 6/(dim_x - 1)

    n_traj = mu.shape[0]
    x_domain = torch.linspace(-3, 3, dim_x, dtype=torch.float64).unsqueeze(0).repeat(n_traj, 1)
    g = torch.exp(-0.5*(x_domain)**2/(scale**2))

    dxdt = torch.zeros_like(x)

    ip1 = (np.arange(dim_x) + 1) % dim_x
    im1 = (np.arange(dim_x) - 1) % dim_x
    dx1 = (x[:, ip1] - x[:, im1])/(2*h)
    dx2 = (x[:, ip1] + x[:, im1] - 2*x[:, :])/(h**2)
    dxdt = visc.unsqueeze(1)*dx2 - x*dx1 + (1 + sigma*f)*g

    # comment out these lines to get periodic boundary conditions
    # uncomment these lines to get dirichlet boundary conditions
    dxdt[:, 0] = 0.0
    dxdt[:, -1] = 0.0

    return dxdt

def kernel(z1: Float[Tensor, "n_tstep"],
           z2: Float[Tensor, "n_tstep"],
           scale: float
) -> Float[Tensor, "n_tstep n_tstep"]:
    dim_z = z1.shape[0]
    z1 = z1.unsqueeze(1).repeat(1, dim_z)
    z2 = z2.unsqueeze(0).repeat(dim_z, 1)
    dist = torch.abs(z1 - z2)
    kern = torch.exp(-dist**2/(2*scale**2))
    return kern

def forcing(t: Float[Tensor, "n_tstep"],
) -> Float[Tensor, "n_tstep dim_f"]:
    n_tstep = t.shape[0]

    K = kernel(t, t, 0.2)
    '''plt.figure()
    plt.imshow(K.numpy())
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "kernel.png"))'''

    eps = torch.min(torch.linalg.eigvals(K).real)
    rootK = torch.linalg.cholesky(K + -2*eps*torch.eye(n_tstep))
    f = rootK.matmul(torch.randn((n_tstep, 1)))

    return f

def create_dataset(mu: Float[Tensor, "n_traj dim_mu"],
                   T: float,
                   n_tstep: int,
                   dim_x: int,
                   train_val_test: str
) -> tuple[Float[Tensor, "n_traj n_tstep"],
           Float[Tensor, "n_traj n_tstep dim_x"],
           Float[Tensor, "n_traj n_tstep dim_x"]
]:
    n_traj = mu.shape[0]
    t = torch.linspace(0.0, T, n_tstep).unsqueeze(0).repeat(n_traj, 1)
    x_domain = torch.linspace(-3, 3, dim_x, dtype=torch.float64)

    x = torch.zeros(n_traj, n_tstep, dim_x)
    f = torch.zeros(n_traj, n_tstep, 1)

    if train_val_test == "train":
        amplitude = np.random.choice([0.75, 0.85], size=n_traj)
        scale = np.random.choice([0.95, 1.05], size=n_traj)
    elif train_val_test == "val":
        amplitude = np.random.choice([0.8], size=n_traj)
        scale = np.random.choice([1.0], size=n_traj)
    elif train_val_test == "test":
        amplitude = np.random.choice([0.8], size=n_traj)
        scale = np.random.choice([1.0], size=n_traj)

    plt.figure(figsize=(12, 3))
    for i in range(n_traj):
        x[i, 0, :] = amplitude[i]*torch.exp(-x_domain**2/scale[i])
        f[i] = forcing(t[i, :])

        if i < 5:
            f_subsampled = forcing(t[i, ::10])
            plt.plot(np.linspace(0, T, n_tstep)[::10], f_subsampled.numpy())
    plt.xlabel("Time")
    plt.ylabel("Forcing")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "forcing.pdf"), format="pdf")

    for i in range(1, n_tstep):
        dt = (t[:, i] - t[:, i - 1]).unsqueeze(1)
        xi1 = x[:, i - 1] + dyn(mu, t[:, i - 1], x[:, i - 1], f[:, i - 1]) * dt
        xi2 = x[:, i - 1] + dyn(mu, t[:, i - 1], xi1, f[:, i - 1]) * dt
        x[:, i] = 0.5 * (xi1 + xi2)
        
    if n_traj == 5:
        for i in range(n_traj):
            plt.figure()
            plt.plot(np.linspace(-3, 3, dim_x), x[i, 0, :].numpy())
            plt.plot(np.linspace(-3, 3, dim_x), x[i, n_tstep//4, :].numpy())
            plt.plot(np.linspace(-3, 3, dim_x), x[i, n_tstep//2, :].numpy())
            plt.plot(np.linspace(-3, 3, dim_x), x[i, n_tstep*3//4, :].numpy())
            plt.plot(np.linspace(-3, 3, dim_x), x[i, -1, :].numpy())
            plt.show()
            plt.savefig(os.path.join(CURR_DIR, f"x{i}.png"))
    
    return t, x, f


def plot_data(data):
    fig, axs = plt.subplots(1, 5, figsize=(12, 3), layout="constrained")
    axs[0].plot(np.linspace(-3, 3, 1000), data["train_x"][0, 0])
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("v")
    axs[0].set_ylim(0, 2)
    axs[0].set_xticks([-3, 0, 3])
    axs[0].set_title(r"t=0")

    axs[1].plot(np.linspace(-3, 3, 1000), data["train_x"][0, 250])
    axs[1].set_xlabel("x")
    axs[1].get_yaxis().set_ticks([])
    axs[1].set_ylim(0, 2)
    axs[1].set_xticks([-3, 0, 3])
    axs[1].set_title(r"t=0.25")

    axs[2].plot(np.linspace(-3, 3, 1000), data["train_x"][0, 500])
    axs[2].set_xlabel("x")
    axs[2].get_yaxis().set_ticks([])
    axs[2].set_ylim(0, 2)
    axs[2].set_xticks([-3, 0, 3])
    axs[2].set_title(r"t=0.5")

    axs[3].plot(np.linspace(-3, 3, 1000), data["train_x"][0, 750])
    axs[3].set_xlabel("x")
    axs[3].get_yaxis().set_ticks([])
    axs[3].set_ylim(0, 2)
    axs[3].set_xticks([-3, 0, 3])
    axs[3].set_title(r"t=0.75")

    axs[4].plot(np.linspace(-3, 3, 1000), data["train_x"][0, 1000])
    axs[4].set_xlabel("x")
    axs[4].get_yaxis().set_ticks([])
    axs[4].set_ylim(0, 2)
    axs[4].set_xticks([-3, 0, 3])
    axs[4].set_title(r"t=1")

    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "data.pdf"))

def main():
    n_traj_train = 10
    n_traj_val = 10
    n_traj_test = 10

    n_tstep = 1001
    every_nth_tstep = 1
    dim_x = 1000

    visc_lb = 0.001
    visc_ub = 0.01
    sigma = 0.2
    gscale = 0.5

    train_T = 1.0
    train_mu = torch.cat([torch.rand(n_traj_train, 1)*(visc_ub - visc_lb) + visc_lb,
                          torch.linspace(sigma, sigma, n_traj_train).unsqueeze(1),
                          torch.linspace(gscale, gscale, n_traj_train).unsqueeze(1)
                          ], dim=1)

    val_T = 1.0
    val_mu = torch.cat([torch.rand(n_traj_val, 1)*(visc_ub - visc_lb) + visc_lb,
                        torch.linspace(sigma, sigma, n_traj_val).unsqueeze(1),
                        torch.linspace(gscale, gscale, n_traj_val).unsqueeze(1)
                        ], dim=1)
    
    test_T = 1.0
    test_mu = torch.cat([torch.rand(n_traj_test, 1)*(visc_ub - visc_lb) + visc_lb,
                         torch.linspace(sigma, sigma, n_traj_test).unsqueeze(1),
                         torch.linspace(gscale, gscale, n_traj_test).unsqueeze(1)
                         ], dim=1)

    train_t, train_x, train_f = create_dataset(train_mu, train_T, n_tstep, dim_x, "train")
    val_t, val_x, val_f = create_dataset(val_mu, val_T, n_tstep, dim_x, "val")
    test_t, test_x, test_f = create_dataset(test_mu, test_T, n_tstep, dim_x, "test")

    print(train_f.shape)
    svdvals_f = torch.linalg.svdvals(train_f[:, :, 0])
    print(svdvals_f.shape)
    cum_energy = torch.cumsum(svdvals_f**2, dim=0)/torch.sum(svdvals_f**2)
    n_modes_99 = torch.sum(cum_energy < 0.99)
    print(n_modes_99)

    print('Any nan?', torch.any(torch.isnan(train_x)).item())

    assert train_mu.shape == (n_traj_train, 3)
    assert train_t.shape == (n_traj_train, n_tstep)
    assert train_x.shape == (n_traj_train, n_tstep, dim_x)
    assert train_f.shape == (n_traj_train, n_tstep, 1)

    assert val_mu.shape == (n_traj_val, 3)
    assert val_t.shape == (n_traj_val, n_tstep)
    assert val_x.shape == (n_traj_val, n_tstep, dim_x)
    assert val_f.shape == (n_traj_val, n_tstep, 1)

    assert test_mu.shape == (n_traj_test, 3)
    assert test_t.shape == (n_traj_test, n_tstep)
    assert test_x.shape == (n_traj_test, n_tstep, dim_x)
    assert test_f.shape == (n_traj_test, n_tstep, 1)

    train_t = train_t[:, ::every_nth_tstep]
    train_x = train_x[:, ::every_nth_tstep]
    train_f = train_f[:, ::every_nth_tstep]

    val_t = val_t[:, ::every_nth_tstep]
    val_x = val_x[:, ::every_nth_tstep]
    val_f = val_f[:, ::every_nth_tstep]

    test_t = test_t[:, ::every_nth_tstep]
    test_x = test_x[:, ::every_nth_tstep]
    test_f = test_f[:, ::every_nth_tstep]

    fig, ax = plt.subplots()
    ax.plot(train_x[0, :, 0].numpy(), train_x[0, :, 1].numpy(), label="train", color="blue")
    ax.plot(val_x[0, :, 0].numpy(), val_x[0, :, 1].numpy(), label="train", color="red")
    plt.show()

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
    
    plot_data(data)

    with open(os.path.join(CURR_DIR, f"data_ll_{n_traj_train}_{n_traj_val}_{n_traj_test}.pkl"), "wb") as f:
        pkl.dump(data, f)

if __name__ == "__main__":
    main()