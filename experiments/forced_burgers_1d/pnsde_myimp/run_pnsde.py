from experiments.forced_burgers_1d.pnsde_myimp.train_pnsde import main as train_pnsde_main
from experiments.forced_burgers_1d.pnsde_myimp.postproc_pnsde import main as postproc_pnsde_main
#from experiments.forced_burgers_1d.pnsde_myimp.make_animation import main as make_animation_main
import pathlib
import sys

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    hparams = {'data_file': "data_bcf_100_10_10_param.pkl",
                'dim_z': 5}
    print(hparams)

    train_pnsde_main(**hparams, overwrite=OVERWRITE)
    postproc_pnsde_main(**hparams)
    #make_animation_main(**hparams)