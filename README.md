# Code

To run:

```sh
python runner.py new --save <SAVE_NAME> --data <PATH_TO_DATA_DIR> --dataset <DATASET> --model <model_name> [options] --n 1000 - train - test - kernel_fit - done
```

Options for Bayesian regression models are:

- gp
- gp_3l
- gp_acos
- gp_acos3
- deep_gp_svi (options: --width 2)
- deep_gp_hmc (options: --width 2)
- deep_gp_3l2l_hmc (options: --width 2)
- deep_gp_3l3l_hmc (options: --width1 2 --width2 2)
- nn_hmc (options: --width 2)
- nn_3l_hmc (options: --width 2)

Options for non-Bayesian classification models are:

- nn (options: --widths 16,32,32)
- lenet (options: --width 256 --conv-width 16)
- resnet (options: --depth 14 --width 64)
