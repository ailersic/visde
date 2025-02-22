import torch
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import visde
from experiments.flowcontrol_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
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
    #n_traj = mu.shape[0]
    n_win = 1
    n_batch = 32
    n_tsteps = t.shape[1]
    chan_x = x.shape[2]

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
    
    model.decoder.dec_var = model.decoder.dec_logvar_mean.exp()
    
    i_traj = 0


    fig = plt.figure(figsize=(20, 7))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(chan_x, 4),
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
        xs = model.decoder.sample(1, mu_i_batch, zs[j]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()
        x_true = x[i_traj].cpu().detach().numpy()

        x_min = np.min(x_true[:, 0])
        x_max = np.max(x_true[:, 0])

        for i in range(chan_x):
            im1 = axgrid[i].imshow(x_true[j, i], cmap=cmap, vmin=x_min, vmax=x_max)
            axgrid[i].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        
        for i in range(chan_x):
            im2 = axgrid[i+chan_x].imshow(x_mean[i], cmap=cmap, vmin=x_min, vmax=x_max)
            axgrid[i+chan_x].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[i+chan_x].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        
        for i in range(chan_x):
            im3 = axgrid[i+2*chan_x].imshow(np.abs(x_true[j, i] - x_mean[i]), cmap='afmhot', vmin=0, vmax=0.22)
            axgrid[i+2*chan_x].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[i+2*chan_x].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        for i in range(chan_x):
            im4 = axgrid[i+3*chan_x].imshow(x_std[i], cmap='afmhot', vmin=0, vmax=0.22)
            axgrid[i+3*chan_x].add_patch(Ellipse((40, 38), 2*0.5/0.05, 2*0.5/0.0525, edgecolor='black', facecolor='white', linewidth=1))
            axgrid[i+3*chan_x].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

        axgrid[0].set_title("True Solution")
        axgrid[chan_x].set_title("Prediction Mean")
        axgrid[2*chan_x].set_title("Absolute Error")
        axgrid[3*chan_x].set_title("Prediction Std. Dev.")
        
        axgrid.cbar_axes[0].colorbar(im1, ticks=[0, 1, 2])
        axgrid.cbar_axes[0].set_xticklabels(["0", "1", "2"])
        axgrid.cbar_axes[1].colorbar(im2, ticks=[0, 1, 2])
        axgrid.cbar_axes[1].set_xticklabels(["0", "1", "2"])
        axgrid.cbar_axes[2].colorbar(im3, ticks=[0, 0.1, 0.2])
        axgrid.cbar_axes[2].set_xticklabels(["0", "0.1", "0.2"])
        axgrid.cbar_axes[3].colorbar(im4, ticks=[0, 0.1, 0.2])
        axgrid.cbar_axes[3].set_xticklabels(["0", "0.1", "0.2"])

        axgrid[0].set_ylabel(r"$x$ vel.")
        axgrid[1].set_ylabel(r"$y$ vel.")
        axgrid[2].set_ylabel(r"pressure")
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_tsteps, interval=30)
    ani.save(filename=os.path.join(OUT_DIR, f"{train_val_test}_pred_vs_true.gif"), writer="pillow")
    fig.show()

if __name__ == "__main__":
    #main("train")
    main("test")