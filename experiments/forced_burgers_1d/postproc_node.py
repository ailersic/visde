import torch
from torch import Tensor
from torchdiffeq import odeint

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

#import visde
from experiments.forced_burgers_1d.train_autoenc import create_autoencoder
from experiments.forced_burgers_1d.train_node import create_node, NODEConfig, TestNODE, ODE

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postproc_node")
DATA_FILE = "data_ll_200_40_50.pkl"

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(train_val_test: str = "test"):
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{train_val_test}_mu"].to(device)
    t = data[f"{train_val_test}_t"].to(device)
    x = data[f"{train_val_test}_x"].to(device)
    f = data[f"{train_val_test}_f"].to(device)

    dim_z = 9
    dim_x = x.shape[-1]
    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 64
    n_tsteps = t.shape[1]

    norm_rmse = np.zeros(n_traj)

    nodeconfig = NODEConfig(lr=1e-3, lr_sched_freq=5000)

    aeconfig, encoder, decoder = create_autoencoder(dim_z, n_win, DATA_FILE)
    drift = create_node(dim_z, DATA_FILE)
    model = TestNODE.load_from_checkpoint("experiments/forced_burgers_1d/logs_node/version_ref_200_ll_2/checkpoints/epoch=49-step=80000.ckpt",
                                                 config=nodeconfig,
                                                 encoder=encoder,
                                                 decoder=decoder,
                                                 drift=drift
                                                 ).to(device)
    model.eval()
    

    tsamples = [0, 249, 499, 749, 999]

    fig = plt.figure(figsize=(6*n_traj, 3*len(tsamples)))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(len(tsamples), 4*n_traj),
                    axes_pad=0.20,
                    share_all=True,
                    direction="column"
                    )
    #fig, axs = plt.subplots(len(tsamples), 3*n_traj, figsize=(3*n_traj, 2*len(tsamples)))
    
    # Initial state y0, the ODE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
        print(f"Integrating ODE for trajectory {train_val_test} {i_traj}...", flush=True)

        mu_i = mu[i_traj].unsqueeze(0)
        mu_i_batch = mu_i.repeat((n_batch, 1))
        t_i = t[i_traj]
        x0_i = x[i_traj, :n_win, :].unsqueeze(0)
        f_i = f[i_traj]

        z0_i, _ = model.encoder(mu_i, x0_i)
        z0_i = z0_i.repeat((n_batch, 1))
        print(z0_i.shape)
        ode = ODE(model.drift, mu_i, t_i, f_i)
        with torch.no_grad():
            zs = odeint(ode, z0_i, t_i)
        print("done", flush=True)

        assert isinstance(zs, Tensor), "zs is expected to be a single tensor"

        sqerr = np.zeros(n_tsteps)
        norm_sqerr = np.zeros(n_tsteps)
        aenc_sqerr = np.zeros(n_tsteps)
        aenc_norm_sqerr = np.zeros(n_tsteps)

        z_i = torch.zeros(n_tsteps, 1, dim_z).to(device)

        print("Decoding trajectory...", flush=True)
        for j_t in range(n_tsteps):
            if j_t % 100 == 0:
                print(f"{j_t}...", flush=True)

            xs, _ = model.decoder(mu_i_batch, zs[j_t])
            x_mean = xs.detach().mean(dim=0)
            x_err = x_mean - x[i_traj, j_t]

            sqerr[j_t] = x_err.pow(2).sum().item()
            norm_sqerr[j_t] = sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()

            z_i[j_t], _ = model.encoder(mu_i, x[i_traj, j_t:(j_t+n_win)].unsqueeze(0))
            x_rec_ij, _ = model.decoder(mu_i, z_i[j_t])
            aenc_err = x_rec_ij - x[i_traj, j_t]

            aenc_sqerr[j_t] = aenc_err.pow(2).sum().item()
            aenc_norm_sqerr[j_t] = aenc_sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()
        print("done", flush=True)

        norm_rmse[i_traj] = np.sqrt(np.mean(norm_sqerr))
        
        print(f"Mean Normalized RMSE: {norm_rmse[i_traj]}", flush=True)

        figrmse, ax = plt.subplots(figsize=(12, 6))
        #ax.plot(rmse, label="RMSE")
        ax.plot(np.sqrt(norm_sqerr), label="Normalized Rel. Error")
        #ax.plot(aenc_rmse, label="AEnc RMSE")
        ax.plot(np.sqrt(aenc_norm_sqerr), label="AEnc Normalized Rel. Error")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"Mean Rel. Error: {np.mean(norm_rmse):.3f}, Max Rel. Error: {np.max(norm_rmse):.3f}")
        ax.legend()
        figrmse.savefig(os.path.join(OUT_DIR, f"{train_val_test}_rmse_traj_{i_traj}.png"))
        figrmse.show()

        figz, axs_z = plt.subplots(dim_z, 1, figsize=(12, 6*dim_z))
        z_enc = z_i.mean(1)
        z_pred = zs.mean(1)
        z_std = zs.std(1)
        for k in range(dim_z):
            axs_z[k].plot(z_enc[:, k].detach().cpu().numpy(), label=f"Encoded true state {k}", linestyle="--", color="blue")
            axs_z[k].plot(z_pred[:, k].detach().cpu().numpy(), label=f"Latent dynamics {k}", linestyle=":", color="red")
            axs_z[k].fill_between(np.arange(n_tsteps), z_pred[:, k].detach().cpu().numpy() - 3*z_std[:, k].detach().cpu().numpy(),
                                  z_pred[:, k].detach().cpu().numpy() + 3*z_std[:, k].detach().cpu().numpy(), color="red", alpha=0.2)
            axs_z[k].legend()
            axs_z[k].set_xlabel("Time step")
            axs_z[k].set_ylabel(f"Latent state {k}")
        figz.savefig(os.path.join(OUT_DIR, f"{train_val_test}_latent_traj_{i_traj}.png"))
        figz.show()

        x_true = x[i_traj].cpu().detach().numpy()

        for j, j_t in enumerate(tsamples):
            xs, _ = model.decoder(mu_i_batch, zs[j_t])
            #xs = model.decoder.sample(1, mu_i_batch, zs[j_t]).detach()
            x_mean = xs.mean(dim=0).cpu().detach().numpy()
            x_std = xs.std(dim=0).cpu().detach().numpy()

            id1 = j + 4*i_traj*len(tsamples)
            id2 = j + (4*i_traj + 1)*len(tsamples)
            id3 = j + (4*i_traj + 2)*len(tsamples)
            id4 = j + (4*i_traj + 3)*len(tsamples)
            
            axgrid[id1].plot(np.linspace(0, 1, dim_x), x_true[j_t])
            axgrid[id1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            axgrid[id2].plot(np.linspace(0, 1, dim_x), x_mean)
            axgrid[id2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            axgrid[id3].plot(np.linspace(0, 1, dim_x), np.abs(x_mean - x_true[j_t]))
            axgrid[id3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            axgrid[id4].plot(np.linspace(0, 1, dim_x), x_std)
            axgrid[id4].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            if j == 0:
                axgrid[id1].set_title("Truth")
                axgrid[id2].set_title("Mean")
                axgrid[id3].set_title("Error")
                axgrid[id4].set_title("Std. Dev.")

    for j, j_t in enumerate(tsamples):
        axgrid[j].set_ylabel(f"$t={t[0, j_t]:.2f}$")
    
    fig.savefig(os.path.join(OUT_DIR, f"{train_val_test}_pred_vs_true.pdf"), format='pdf')
    fig.show()

    print(f"Normalized RMSE Mean: {np.mean(norm_rmse)}, Std Dev: {np.std(norm_rmse)}", flush=True)

if __name__ == "__main__":
    #main("train")
    main("test")