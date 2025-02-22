import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import visde
from experiments.flowcontrol_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 14})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postproc_visde")

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(train_val_test: str = "test"):
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{train_val_test}_mu"].to(device)
    t = data[f"{train_val_test}_t"].to(device)
    x = data[f"{train_val_test}_x"].to(device)
    f = data[f"{train_val_test}_f"].to(device)

    dim_z = 9
    n_traj = 1#mu.shape[0]
    n_win = 1
    n_batch = 32
    n_dec_samp = 64
    n_tsteps = t.shape[1]
    chan = 2

    norm_rmse = np.zeros(n_traj)

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win)
    model = visde.LatentSDE.load_from_checkpoint("experiments/flowcontrol_2d/logs_visde/version_ref_4/checkpoints/epoch=499-step=150000.ckpt",
                                                 config=dummy_model.config,
                                                 encoder=dummy_model.encoder,
                                                 decoder=dummy_model.decoder,
                                                 drift=dummy_model.drift,
                                                 dispersion=dummy_model.dispersion,
                                                 loglikelihood=dummy_model.loglikelihood,
                                                 latentvar=dummy_model.latentvar).to(device)
    model.eval()
    model.encoder.resample_params()
    model.decoder.resample_params()
    model.drift.resample_params()
    model.dispersion.resample_params()
    
    model.dispersion.disp = torch.exp(model.dispersion.logdisp_mean)
    model.decoder.dec_var = torch.exp(model.decoder.dec_logvar_mean)

    print([p for p in model.dispersion.parameters()])

    tsamples = [0, 33, 66, n_tsteps - 1]

    fig = plt.figure(figsize=(6.3, 3.3))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(len(tsamples) + 1, 3*n_traj),
                    axes_pad=0.05,
                    share_all=True,
                    #label_mode="1",
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_pad=0.10,
                    cbar_size="25%",
                    direction="column"
                    )
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
        print(f"Integrating SDE for trajectory {train_val_test} {i_traj}...", flush=True)

        mu_i = mu[i_traj].unsqueeze(0)
        mu_i_batch = mu_i.repeat((n_batch, 1))
        t_i = t[i_traj]
        x0_i = x[i_traj, :n_win, :].unsqueeze(0)
        f_i = f[i_traj]

        z0_i = model.encoder.sample(n_batch, mu_i, x0_i)
        sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
        with torch.no_grad():
            zs = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
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

            xs = model.decoder.sample(n_dec_samp, mu_i_batch, zs[j_t]).detach()
            x_mean = xs.mean(dim=0)
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
        ax.plot(np.sqrt(norm_sqerr), label=r"Latent dynamics $\varepsilon$")
        ax.plot(np.sqrt(aenc_norm_sqerr), label=r"Autoencoder $\varepsilon$")
        ax.set_xlabel("Time step")
        ax.set_ylabel(r"$\varepsilon$")
        ax.set_title(f"RMSRE: {norm_rmse[i_traj]:.3f}")
        ax.legend()
        figrmse.savefig(os.path.join(OUT_DIR, f"{train_val_test}_rmsre_traj_{i_traj}.png"))
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

        cmap = "coolwarm"
        for j, j_t in enumerate(tsamples):
            xs = model.decoder.sample(n_dec_samp, mu_i_batch, zs[j_t]).detach()
            x_mean = xs.mean(dim=0).cpu().detach().numpy()
            x_std = xs.std(dim=0).cpu().detach().numpy()
            x_true = x[i_traj].cpu().detach().numpy()

            x_min = np.min(x_true[:, chan])
            x_max = np.max(x_true[:, chan])

            id1 = j + 3*i_traj*(len(tsamples) + 1)
            id2 = j + (3*i_traj + 1)*(len(tsamples) + 1)
            id3 = j + (3*i_traj + 2)*(len(tsamples) + 1)
            if j == 3:
                axgrid[id1].add_patch(Ellipse((220, 70), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id1].add_patch(Ellipse((220, 40), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id1].add_patch(Ellipse((220, 10), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id1].axis('off')

                axgrid[id2].add_patch(Ellipse((220, 70), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id2].add_patch(Ellipse((220, 40), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id2].add_patch(Ellipse((220, 10), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id2].axis('off')

                axgrid[id3].add_patch(Ellipse((220, 70), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id3].add_patch(Ellipse((220, 40), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id3].add_patch(Ellipse((220, 10), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
                axgrid[id3].axis('off')
                
                id1 = j + 3*i_traj*(len(tsamples) + 1) + 1
                id2 = j + (3*i_traj + 1)*(len(tsamples) + 1) + 1
                id3 = j + (3*i_traj + 2)*(len(tsamples) + 1) + 1

            im1 = axgrid[id1].imshow(x_true[j_t, chan], cmap=cmap, vmin=x_min, vmax=x_max)
            axgrid[id1].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[id1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            im2 = axgrid[id2].imshow(x_mean[chan], cmap=cmap, vmin=x_min, vmax=x_max)
            axgrid[id2].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[id2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            im3 = axgrid[id3].imshow(x_std[chan], cmap='afmhot', vmin=0.04, vmax=0.16)
            axgrid[id3].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[id3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            if j == 0:
                axgrid[id1].set_title("True\nSolution")
                axgrid[id2].set_title("Prediction\nMean")
                axgrid[id3].set_title("Prediction\nStd. Dev.")
            elif j == len(tsamples) - 1:
                axgrid.cbar_axes[i_traj].colorbar(im1, ticks=[-2, -1, 0, 1, 2])
                axgrid.cbar_axes[i_traj].set_xticklabels(["-2", "-1", "0", "1", "2"])
                axgrid.cbar_axes[i_traj+1].colorbar(im2, ticks=[-2, -1, 0, 1, 2])
                axgrid.cbar_axes[i_traj+1].set_xticklabels(["-2", "-1", "0", "1", "2"])
                axgrid.cbar_axes[i_traj+2].colorbar(im3, ticks=[0.05, 0.1, 0.15])
                axgrid.cbar_axes[i_traj+2].set_xticklabels(["0.05", "0.1", "0.15"])

    axgrid[0].set_ylabel(r"$0$")
    axgrid[1].set_ylabel(r"$\frac{1}{2}T_v$")
    axgrid[2].set_ylabel(r"$T_v$")
    axgrid[4].set_ylabel(r"$5T_v$")

    axgrid[2].set_title(r"Time $t$", rotation='vertical',x=-0.3,y=-0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{train_val_test}_pred_vs_true_chan_{chan}.pdf"), format='pdf')
    fig.show()

    print(f"Normalized RMSE Mean: {np.mean(norm_rmse)}, Std Dev: {np.std(norm_rmse)}", flush=True)

if __name__ == "__main__":
    #main("train")
    main("test")