# Preparing the Burgers dataset

To prepare this dataset, simply run the script `gen_data.py`. This will create a file `data_#1_#2_#3.pkl`, where #1 is the number of training trajectories, #2 is validation, and #3 is test. You are now ready to begin training.

To train the PNODE/PNSDE models, first run the script `forcing_to_param.py` to convert the time-dependent forcing in the dataset to time-independent parameters. This will create a new file called `data_#1_#2_#3_param.pkl`.