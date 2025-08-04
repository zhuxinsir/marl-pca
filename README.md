# Code for PCA

This instruction hosts the PyTorch implementation of PCA accompanying the paper ''**Proximal Credit Assignment for Cooperative Multi-Agent Reinforcement Learning in Partially Observable Games**''. PCA is a neighbor-based credit assignment method that factorizes and simplifies the complex interaction analysis of multi-agent systems into a sequential evaluation process for more efficient learning.

The implementation is based on the [PyMARL](https://github.com/oxwhirl/pymarl) framework and [SMAC](https://github.com/oxwhirl/smac). All of our SMAC experiments are based on the latest PyMARL utilizing SC2.4.6.10. The underlying dynamics are sufficiently different, so you cannot compare runs across various versions. 


## Setup

Set up the working environment: 

```shell
conda create -n marl python=3.8 -y
conda activate marl

bash install_dependecies.sh
```

Set up the StarCraftII game core (SC2.4.6.10): 

```shell
bash install_sc2.sh
```


## Training

To train `PCA` on the `2s3z` scenario, 

```shell
python3 src/main.py --config=pca --env-config=sc2 with env_args.map_name=2s3z
```

Change the `map_name` accordingly for other scenarios (e.g. `1c3s5z`). All results will be saved in the `results` folder. 

The config file `src/config/algs/pca.yaml` contains default hyperparameters for PCA.


## Evaluation

### TensorBoard

One could set `use_tensorboard` to `True` in `src/config/default.yaml`, and the training tensorboards will be saved in the `results/tb_logs` directory, containing useful info such as test battle win rate during training. 

### Saving models

Same as PyMARL, set `save_model` to `True` in `src/config/default.yaml` and the learn model during training will be saved in `results/models/` directory. The frequency for saving models can be adjusted by setting the parameter `save_model_interval`.

### Loading models

Saved models can be loaded by adjusting the `checkpoint_path` parameter in `src/config/default.yaml`. For instance, to load model under path `result/model/[timesteps]/agent.th`, set `checkpoint_path` to `result/model/[timesteps]`.

### Saving Starcraft II Replay

The learned model loaded from `checkpoint_path` can be evaluated by setting `evaluate` to `True` in `src/config/default.yaml`. To save the Starcraft II replays, please make sure configure `save_replay` is set to `True`, and use the `episode_runner`.

Check out [PyMARL](https://github.com/oxwhirl/pymarl) documentation for more information.

## See Also

See [SMAC](https://github.com/oxwhirl/smac) and [PyMARL](https://github.com/oxwhirl/pymarl) for additional instructions.
