import torch
from torch import Tensor
from torch.utils.data import DataLoader
from jaxtyping import Float

import visde
# ruff: noqa: F821, F722

def create_exp_dataset(mu_samples: Float[Tensor, "n_traj dim_mu"],
                    t_samples: Float[Tensor, "n_tstep"],
                    dim_x: int
) -> tuple[Float[Tensor, "n_traj dim_mu"],
            Float[Tensor, "n_traj n_tstep"],
            Float[Tensor, "n_traj n_tstep dim_x"],
            Float[Tensor, "n_traj n_tstep dim_f"]
]:
    n_tstep = t_samples.shape[0]
    n_traj = mu_samples.shape[0]

    x = torch.zeros(n_traj, n_tstep, dim_x)
    x0 = torch.ones(dim_x)
    f = torch.zeros(n_traj, n_tstep, 1)

    for i in range(n_traj):
        mu_mat = torch.diag(mu_samples[i, :-1])
        for j in range(n_tstep):
            x[i, j] = torch.matrix_exp(mu_mat * t_samples[j]) @ x0
    
    t = t_samples.unsqueeze(0).repeat(n_traj, 1)
    return mu_samples, t, x, f

def create_exp_dataloader(dim_mu, dim_x, n_win, n_batch) -> DataLoader:
    t_samples = torch.linspace(0, 1, 101)
    n_traj = 5
    mu_samples = torch.linspace(1, 2, n_traj).unsqueeze(1).repeat(1, dim_mu)
    assert dim_mu-1 == dim_x, "dim mu must be dim x + 1"
    train_mu, train_t, train_x, train_f = create_exp_dataset(mu_samples, t_samples, dim_x)
    print(train_mu.shape, train_t.shape, train_x.shape, train_f.shape)

    dataset = visde.MultiEvenlySpacedTensors(train_mu, train_t, train_x, train_f, n_win)
    sampler = visde.MultiTemporalSampler(dataset, n_batch, n_repeats=1)
    train_loader = DataLoader(
        dataset,
        num_workers=6,
        persistent_workers=True,
        batch_sampler=sampler,
    )

    return train_loader

def test_dataloader():
    dataloader = create_exp_dataloader(5, 4, 3, 2)

    for mu, t, x_win, x, f in dataloader:
        print(mu.shape, t.shape, x_win.shape, x.shape, f.shape)
        print(t)

if __name__ == "__main__":
    test_dataloader()