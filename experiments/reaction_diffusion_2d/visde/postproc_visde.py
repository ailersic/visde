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
import warnings

import visde
from experiments.reaction_diffusion_2d.visde.def_model import create_latent_sde
from datasets.reaction_diffusion_2d.load_data import load_data

plt.rcParams.update({'font.size': 14})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def plot_rel_err(traj_dir: str, rel_err: np.ndarray, aenc_rel_err: np.ndarray) -> None:
    figerr, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rel_err, label="Latent Model")
    ax.plot(aenc_rel_err, label="Autoencoder")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Relative Error")
    ax.set_title(f"Mean Relative Error: {np.mean(rel_err):.5f}")
    ax.legend()
    figerr.savefig(os.path.join(traj_dir, "error.png"))
    figerr.show()
    plt.close(figerr)

def plot_latent_state(traj_dir: str, z_enc: np.ndarray, z_int: np.ndarray, dim_z: int) -> None:
    figz, axs_z = plt.subplots(dim_z, 1, figsize=(12, 6*dim_z))
    z_enc_m = z_enc.mean(1)
    z_pred = z_int.mean(1)
    z_std = z_int.std(1)
    n_tsteps = z_enc.shape[0]
    for k in range(dim_z):
        axs_z[k].plot(z_enc_m[:, k], label=f"Encoded true state {k}", linestyle="--", color="blue")
        axs_z[k].plot(z_pred[:, k], label=f"Latent dynamics {k}", linestyle=":", color="red")
        axs_z[k].fill_between(np.arange(n_tsteps), z_pred[:, k] - z_std[:, k],
                                z_pred[:, k] + z_std[:, k], color="red", alpha=0.05)
        axs_z[k].fill_between(np.arange(n_tsteps), z_pred[:, k] - 2*z_std[:, k],
                                z_pred[:, k] + 2*z_std[:, k], color="red", alpha=0.05)
        axs_z[k].fill_between(np.arange(n_tsteps), z_pred[:, k] - 3*z_std[:, k],
                                z_pred[:, k] + 3*z_std[:, k], color="red", alpha=0.05)
        axs_z[k].legend()
        axs_z[k].set_xlabel("Time step")
        axs_z[k].set_ylabel(f"Latent state {k}")
    figz.savefig(os.path.join(traj_dir, f"latent.png"))
    figz.show()
    plt.close(figz)

def plot_pred_state(traj_dir: str, x_true: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, tsamples: list, t_i: np.ndarray) -> None:
    fig = plt.figure(figsize=(6.3, 7))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(len(tsamples) + 1, 4),
                    axes_pad=0.05,
                    share_all=True,
                    #label_mode="1",
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_pad=0.10,
                    cbar_size="25%",
                    direction="column"
                    )

    cmap = "coolwarm"#"nipy_spectral"
    chan = 0
    
    for j, j_t in enumerate(tsamples):
        x_min = np.min(x_true[:, chan])
        x_max = np.max(x_true[:, chan])

        if j < len(tsamples) - 1:
            id1 = j
            id2 = j + (len(tsamples) + 1)
            id3 = j + 2*(len(tsamples) + 1)
            id4 = j + 3*(len(tsamples) + 1)

            axgrid[j].set_ylabel(f"{t_i[tsamples[j]]:.1f}")
        else:
            id1 = j + 1
            id2 = j + (len(tsamples) + 1) + 1
            id3 = j + 2*(len(tsamples) + 1) + 1
            id4 = j + 3*(len(tsamples) + 1) + 1

            axgrid[id1 - 1].add_patch(Ellipse((50, 80), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id1 - 1].add_patch(Ellipse((50, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id1 - 1].add_patch(Ellipse((50, 20), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id1 - 1].axis('off')

            axgrid[id2 - 1].add_patch(Ellipse((50, 80), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id2 - 1].add_patch(Ellipse((50, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id2 - 1].add_patch(Ellipse((50, 20), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id2 - 1].axis('off')

            axgrid[id3 - 1].add_patch(Ellipse((50, 80), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id3 - 1].add_patch(Ellipse((50, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id3 - 1].add_patch(Ellipse((50, 20), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id3 - 1].axis('off')

            axgrid[id4 - 1].add_patch(Ellipse((50, 80), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id4 - 1].add_patch(Ellipse((50, 50), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id4 - 1].add_patch(Ellipse((50, 20), 10, 10, edgecolor='black', facecolor='black', linewidth=1))
            axgrid[id4 - 1].axis('off')

            axgrid[j + 1].set_ylabel(f"{t_i[tsamples[j]]:.1f}")

        im1 = axgrid[id1].imshow(x_true[j_t, chan], cmap=cmap, vmin=x_min, vmax=x_max)
        axgrid[id1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        
        im2 = axgrid[id2].imshow(x_mean[j_t, chan], cmap=cmap, vmin=x_min, vmax=x_max)
        axgrid[id2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        im3 = axgrid[id3].imshow(np.abs(x_true[j_t, chan] - x_mean[j_t, chan]), cmap='afmhot', vmin=0.0, vmax=0.06)
        axgrid[id3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        im4 = axgrid[id4].imshow(x_std[j_t, chan], cmap='afmhot', vmin=0.0, vmax=0.06)
        axgrid[id4].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        if j == 0:
            axgrid[id1].set_title("True\nSolution")
            axgrid[id2].set_title("Prediction\nMean")
            axgrid[id3].set_title("Absolute\nError")
            axgrid[id4].set_title("Prediction\nStd. Dev.")
        elif j == len(tsamples) - 1:
            axgrid.cbar_axes[0].colorbar(im1, ticks=[-1, -0.5, 0, 0.5, 1])
            axgrid.cbar_axes[0].set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
            axgrid.cbar_axes[1].colorbar(im2, ticks=[-1, -0.5, 0, 0.5, 1])
            axgrid.cbar_axes[1].set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
            axgrid.cbar_axes[2].colorbar(im3, ticks=[0.01, 0.03, 0.05])
            axgrid.cbar_axes[2].set_xticklabels([".01", ".03", ".05"])
            axgrid.cbar_axes[3].colorbar(im4, ticks=[0.01, 0.03, 0.05])
            axgrid.cbar_axes[3].set_xticklabels([".01", ".03", ".05"])

    #fig.tight_layout()
    fig.savefig(os.path.join(traj_dir, "pred_vs_true.pdf"), format='pdf')
    fig.show()
    plt.close(fig)

def plot_error_dist(out_dir: str, rel_err: np.ndarray, t_plot: np.ndarray) -> None:
    figerr, ax = plt.subplots(figsize=(6.3, 3))
    n_t = t_plot.shape[0]

    periods = [0, n_t//4, n_t//2, 3*n_t//4, n_t-1]

    ax.plot(t_plot, np.mean(rel_err, axis=0))
    #ax.errorbar(t_plot[periods], np.mean(rel_err, axis=0)[periods], yerr=np.std(rel_err, axis=0)[periods], fmt='o', capsize=5)
    #ax.boxplot(rel_err[:, periods], positions=t_plot[periods], showfliers=False)

    ax.set_xticks(t_plot[periods])
    #ax.set_xticklabels(["0", r"$\frac{1}{2}T_v$", r"$T_v$", r"$\frac{3}{2}T_v$", r"$2T_v$"])
    ax.set_xlabel("Time")
    #ax.set_xlim([t_plot[0]-0.1, t_plot[-1]+0.1])

    #ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_ylabel(r"$\|\bar{u}(t) - u(t)\|/\|u(t)\|$")
    #ax.set_ylim([0, np.max(rel_err[:, periods])*1.1])
    
    ax.grid()
    figerr.tight_layout()
    figerr.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_err_dist.pdf"), format='pdf')
    figerr.show()
    plt.close(figerr)

def main(data_file: str = "data_noisy.pkl",
        dim_z = 2,
        n_epochs: int = 1000,
        lr: float = 3e-3,
        lr_sched_freq: int = 2000,
) -> None:
    data = load_data(data_file)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    threshold = False

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 32
    n_batch_decoder = 64
    n_tsteps = t.shape[1]
    n_chan = x.shape[2]

    shape_x = x.shape[2:]

    rel_err = np.zeros((n_traj, n_tsteps))
    aenc_rel_err = np.zeros((n_traj, n_tsteps))

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win, data_file, lr, lr_sched_freq, device)
    version = "_".join([data_file.split(".")[0], str(dim_z), "128", "256", str(n_epochs), str(lr), str(lr_sched_freq)])
    ckpt_dir = os.path.join(CURR_DIR, "logs_visde", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "postproc_visde", version)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            ckpt_file = file
    
    model = visde.LatentSDE.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file),
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
    
    #model.dispersion.disp = torch.exp(model.dispersion.logdisp_mean)
    #model.decoder.dec_var = torch.exp(model.decoder.dec_logvar_mean)

    #print([p for p in model.dispersion.parameters()])

    model.decoder.dec_var = torch.exp(model.decoder.dec_logvar_mean)
    print([p for p in model.drift.parameters()])
    print([p for p in model.dispersion.parameters()])
    
    if threshold:
        with torch.no_grad():
            max_weight = torch.max(torch.abs(model.drift.net.net[0].weight))
            for i in range(model.drift.net.net[0].weight.shape[0]):
                for j in range(model.drift.net.net[0].weight.shape[1]):
                    if torch.abs(model.drift.net.net[0].weight[i, j]) < 0.05*max_weight:
                        model.drift.net.net[0].weight[i, j] = 0.0
    print(model.drift.net.net[0].weight)

    tsamples = [0, n_tsteps//5, n_tsteps-1]

    print("---")
    for i_traj in range(n_traj):
        print(f"Trajectory {i_traj+1}/{n_traj} from {TRAIN_VAL_TEST} set", flush=True)

        traj_dir = os.path.join(out_dir, f"{TRAIN_VAL_TEST}_traj_{i_traj}")
        pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)

        print(f"Integrating SDE...", flush=True, end="")

        mu_i = mu[i_traj].unsqueeze(0)
        mu_i_batch = mu_i.repeat((n_batch, 1))
        t_i = t[i_traj]
        x0_i = x[i_traj, :n_win].unsqueeze(0)
        f_i = f[i_traj]

        z0_i = model.encoder.sample(n_batch, mu_i, x0_i)
        sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
        # suppress warnings and no gradient tracking
        with warnings.catch_warnings() as X_, torch.no_grad() as Y_:
            warnings.simplefilter("ignore")
            z_int = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
        print("done", flush=True)

        assert isinstance(z_int, Tensor), "zs is expected to be a single tensor"

        z_enc = torch.zeros(n_tsteps, 1, dim_z).to(device)
        #z_int.shape = (n_tsteps, n_batch, dim_z)

        x_true = x[i_traj]
        x_mean = torch.zeros_like(x_true).to(device)
        x_std = torch.zeros_like(x_true).to(device)

        print("Decoding trajectory...", flush=True, end="")
        for j_t in range(n_tsteps):
            if j_t % (n_tsteps//4) == 0:
                print(f"{j_t}...", flush=True, end="")

            xs = model.decoder.sample(n_batch_decoder, mu_i_batch, z_int[j_t]).detach()
            x_mean[j_t] = xs.mean(dim=0)
            x_std[j_t] = xs.std(dim=0)

            rel_err[i_traj, j_t] = ((x_mean[j_t] - x[i_traj, j_t]).pow(2).sum() / x[i_traj, j_t].pow(2).sum()).sqrt().item()

            z_enc[j_t], _ = model.encoder(mu_i, x[i_traj, j_t:(j_t+n_win)].unsqueeze(0))
            x_rec_ij, _ = model.decoder(mu_i, z_enc[j_t])

            aenc_rel_err[i_traj, j_t] = ((x_rec_ij - x[i_traj, j_t]).pow(2).sum() / x[i_traj, j_t].pow(2).sum()).sqrt().item()
        print("done", flush=True)
        
        print(f"Mean relative error: {np.mean(rel_err[i_traj])}", flush=True)

        print("Plotting...", flush=True, end="")

        plot_rel_err(traj_dir,
                    rel_err[i_traj],
                    aenc_rel_err[i_traj])

        plot_latent_state(traj_dir,
                        z_enc.cpu().detach().numpy(),
                        z_int.cpu().detach().numpy(),
                        dim_z)

        plot_pred_state(traj_dir,
                        x_true.cpu().detach().numpy(),
                        x_mean.cpu().detach().numpy(),
                        x_std.cpu().detach().numpy(),
                        tsamples,
                        t[i_traj].cpu().detach().numpy())
        
        print("done", flush=True)
        print("---", flush=True)
    
    plot_error_dist(out_dir,
                    rel_err,
                    t[i_traj].cpu().detach().numpy())

    print(f"Mean relative error: {np.mean(rel_err.flatten())}, Std Dev: {np.std(rel_err.flatten())}", flush=True)

    np.set_printoptions(threshold=np.inf)
    with open(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_error.txt"), "w") as f:
        f.write(np.array2string(rel_err, precision=5))
        f.write("\n")
        f.write(f"Mean: {np.mean(rel_err.flatten()):.5f}, Std Dev: {np.std(rel_err.flatten()):.5f}\n")
    
    print("All done!", flush=True)

if __name__ == "__main__":
    main()
