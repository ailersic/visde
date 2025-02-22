import torch
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import visde
from experiments.reaction_diffusion_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postproc_visde")
DATA_FILE = "data_noisy.pkl"

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

    dim_z = 2
    n_win = 1
    n_batch = 128
    n_dec_samp = 128
    n_tsteps = t.shape[1]
    chan = 0

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win, DATA_FILE)
    model = visde.LatentSDE.load_from_checkpoint("experiments/reaction_diffusion_2d/logs_visde/version_4/checkpoints/epoch=999-step=63000.ckpt",
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
    
    model.decoder.dec_var = model.decoder.dec_logvar_mean.exp()
    
    i_traj = 0

    fig = plt.figure(figsize=(20, 7))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.20,
                    share_all=True,
                    #label_mode="1",
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_pad=0.25,
                    cbar_size="15%",
                    direction="column"
                    )

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

    cmap = "coolwarm"#"nipy_spectral"
    
    def update(j):
        if j % 10 == 0:
            print(f"Frame {j}/{n_tsteps}", flush=True)
        # j is time index
        xs = model.decoder.sample(n_dec_samp, mu_i_batch, zs[j]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()
        x_true = x[i_traj].cpu().detach().numpy()

        x_min = np.min(x_true[:, 0])
        x_max = np.max(x_true[:, 0])

        im1 = axgrid[0].imshow(x_true[j, chan], cmap=cmap, vmin=x_min, vmax=x_max)
        axgrid[0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    
        im2 = axgrid[1].imshow(x_mean[chan], cmap=cmap, vmin=x_min, vmax=x_max)
        axgrid[1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    
        im3 = axgrid[2].imshow(np.abs(x_true[j, chan] - x_mean[chan]), cmap='afmhot', vmin=0, vmax=0.06)
        axgrid[2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        im4 = axgrid[3].imshow(x_std[chan], cmap='afmhot', vmin=0, vmax=0.06)
        axgrid[3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        axgrid[0].set_title("True Solution")
        axgrid[1].set_title("Prediction Mean")
        axgrid[2].set_title("Absolute Error")
        axgrid[3].set_title("Prediction Std. Dev.")
        
        axgrid.cbar_axes[0].colorbar(im1, ticks=[-0.5, 0, 0.5])
        axgrid.cbar_axes[0].set_xticklabels(["-0.5", "0", "0.5"])
        axgrid.cbar_axes[1].colorbar(im2, ticks=[-0.5, 0, 0.5])
        axgrid.cbar_axes[1].set_xticklabels(["-0.5", "0", "0.5"])
        axgrid.cbar_axes[2].colorbar(im3, ticks=[0.01, 0.03, 0.05])
        axgrid.cbar_axes[2].set_xticklabels(["0.01", "0.03", "0.05"])
        axgrid.cbar_axes[3].colorbar(im4, ticks=[0.01, 0.03, 0.05])
        axgrid.cbar_axes[3].set_xticklabels(["0.01", "0.03", "0.05"])
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_tsteps, interval=30)
    ani.save(filename=os.path.join(OUT_DIR, f"{train_val_test}_pred_vs_true_{DATA_FILE}.gif"), writer="pillow")
    fig.show()

if __name__ == "__main__":
    #main("train")
    main("test")