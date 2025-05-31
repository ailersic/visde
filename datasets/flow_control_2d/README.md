# Preparing the flow control dataset

To generate the dataset for this test case, we use the implementation [provided on Github](https://github.com/jerabaul29/Cylinder2DFlowControlDRL) by Rabault et al. Once that is set up, to generate a single trajectory with random forcing, place the script `single_openloop_runner.py` in the directory `Cylinder2DFlowControlDRL/Cylinder2DFlowControlWithRL/baseline_flow` and run it.

Choose a root directory for the dataset and place the `.pvd`/`.vtu` files and `test_strategy.csv` for that trajectory in a subdirectory titled `results_#`, where `#` increments up from 0 for each additional trajectory. For each, using Paraview, resample the velocity/pressure fields onto regular 440 by 80 grids, and save them as `resample_u` and `resample_p` respectively. Finally, place the script `assemble_data.py` in the dataset root directory.

The dataset directory tree should look like this:

- flow_dataset
    - assemble_data.py
    - results_0
        - area_out.pvd
        - u_out.pvd
        - p_out.pvd
        - resample_u.pvd
        - resample_p.pvd
        - test_strategy.csv
        - ...
    - results_1
    - results_2
    - ...

Now run the script `assemble_data.py` and a new file `data.pkl` should appear in the root directory. place that in `visde/datasets/flow_control_2d` and you are now ready to begin training.

To train the PNODE/PNSDE models, first run the script `forcing_to_param.py` to convert the time-dependent forcing in the dataset to time-independent parameters. This will create a new file called `data_param.pkl`.