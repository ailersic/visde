import os
import pathlib
import pickle as pkl

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def load_data(data_file: str = "data.pkl"):
    with open(os.path.join(CURR_DIR, data_file), "rb") as f:
        data = pkl.load(f)
    return data
