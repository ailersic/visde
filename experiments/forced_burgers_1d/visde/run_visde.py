from experiments.forced_burgers_1d.visde.train_visde import main as train_visde_main
from experiments.forced_burgers_1d.visde.postproc_visde import main as postproc_visde_main
from experiments.forced_burgers_1d.visde.make_animation import main as make_animation_main
import pathlib
import sys

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    hparams = {'data_file': "data_bcf_100_10_10.pkl",
                'dim_z': 5}
    print(hparams)

    train_visde_main(**hparams, overwrite=OVERWRITE)
    postproc_visde_main(**hparams)
    #make_animation_main(**hparams)