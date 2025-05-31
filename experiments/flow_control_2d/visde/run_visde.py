from experiments.flow_control_2d.visde.train_visde import main as train_visde_main
from experiments.flow_control_2d.visde.postproc_visde import main as postproc_visde_main
#from experiments.flow_control_2d.visde.make_animation import main as make_animation_main
import pathlib
import sys

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    hparams = {'data_file': "data.pkl",
                'dim_z': 3,
                'n_epochs': 200,
                'lr': 1e-3,
                'lr_sched_freq': 2000,
                }
    print(hparams)

    train_visde_main(**hparams, overwrite=OVERWRITE)
    postproc_visde_main(**hparams)
    #make_animation_main(**hparams)