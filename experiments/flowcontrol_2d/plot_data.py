from matplotlib.patches import Ellipse
import torch
import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def plot_data():
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)

    fig, axs = plt.subplots(7, 3, figsize=(14, 7), layout="constrained")

    axs[0,0].imshow(data["train_x"][0, 0, 0], cmap="coolwarm")
    axs[0,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[0,0].set_title(r"$x$ Velocity")
    axs[0,0].set_ylabel(r"$t=0$")

    axs[0,1].imshow(data["train_x"][0, 0, 1], cmap="coolwarm")
    axs[0,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[0,1].set_title(r"$y$ Velocity")

    axs[0,2].imshow(data["train_x"][0, 0, 2], cmap="coolwarm")
    axs[0,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[0,2].set_title(r"Pressure")

    #

    axs[1,0].imshow(data["train_x"][0, 16, 0], cmap="coolwarm")
    axs[1,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[1,0].set_ylabel(r"$t=\frac{1}{4}T_v$")

    axs[1,1].imshow(data["train_x"][0, 16, 1], cmap="coolwarm")
    axs[1,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    axs[1,2].imshow(data["train_x"][0, 16, 2], cmap="coolwarm")
    axs[1,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    #

    axs[2,0].imshow(data["train_x"][0, 33, 0], cmap="coolwarm")
    axs[2,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[2,0].set_ylabel(r"$t=\frac{1}{2}T_v$")

    axs[2,1].imshow(data["train_x"][0, 33, 1], cmap="coolwarm")
    axs[2,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    axs[2,2].imshow(data["train_x"][0, 33, 2], cmap="coolwarm")
    axs[2,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    #

    axs[3,0].imshow(data["train_x"][0, 49, 0], cmap="coolwarm")
    axs[3,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[3,0].set_ylabel(r"$t=\frac{3}{4}T_v$")

    axs[3,1].imshow(data["train_x"][0, 49, 1], cmap="coolwarm")
    axs[3,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    axs[3,2].imshow(data["train_x"][0, 49, 2], cmap="coolwarm")
    axs[3,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    #

    axs[4,0].imshow(data["test_x"][0, 66, 0], cmap="coolwarm")
    axs[4,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[4,0].set_ylabel(r"$t=T_v$")

    axs[4,1].imshow(data["test_x"][0, 66, 1], cmap="coolwarm")
    axs[4,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    axs[4,2].imshow(data["test_x"][0, 66, 2], cmap="coolwarm")
    axs[4,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    #

    axs[5,0].imshow(torch.zeros_like(data["train_x"][0, 0, 0]), cmap="binary")
    axs[5,0].add_patch(Ellipse((220, 10), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,0].add_patch(Ellipse((220, 40), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,0].add_patch(Ellipse((220, 70), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,0].axis('off')

    axs[5,1].imshow(torch.zeros_like(data["train_x"][0, 0, 0]), cmap="binary")
    axs[5,1].add_patch(Ellipse((220, 10), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,1].add_patch(Ellipse((220, 40), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,1].add_patch(Ellipse((220, 70), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,1].axis('off')

    axs[5,2].imshow(torch.zeros_like(data["train_x"][0, 0, 0]), cmap="binary")
    axs[5,2].add_patch(Ellipse((220, 10), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,2].add_patch(Ellipse((220, 40), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,2].add_patch(Ellipse((220, 70), 15, 15, edgecolor='black', facecolor='black', linewidth=1))
    axs[5,2].axis('off')
    
    #
    
    im0 = axs[6,0].imshow(data["test_x"][0, -1, 0], cmap="coolwarm")
    axs[6,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    axs[6,0].set_ylabel(r"$t=5T_v$")

    im1 = axs[6,1].imshow(data["test_x"][0, -1, 1], cmap="coolwarm")
    axs[6,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    im2 = axs[6,2].imshow(data["test_x"][0, -1, 2], cmap="coolwarm")
    axs[6,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

    fig.colorbar(im0, ax=axs[6,0], location="bottom")
    fig.colorbar(im1, ax=axs[6,1], location="bottom")
    fig.colorbar(im2, ax=axs[6,2], location="bottom")

    #plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "data.pdf"))

if __name__ == "__main__":
    plot_data()