# QnRL: Quantum-native Reinforcement Learning

## Installation

The codebase is provided as an installable Python package called `qnrl`. To install the package via `pip`, you can run:

```bash
# Navigate to `qnrl` source folder.
$ cd path/to/qnrl/

# Install `qnrl` package.
$ python -m pip install .
```

You can verify the package was successfully install by running:

```bash
$ python -c "import importlib.metadata; version=importlib.metadata.version('qnrl'); print(version)"
1.0.0
```

### Requirements

See [`pyproject.toml`](./pyproject.toml) for package requirements. Package management can be done via `pip` or `uv` (see [`uv.lock`](./uv.lock)).

## CLI

We have developed a lightweight command-line interface (CLI) that can be invoked via the Python interpreter like so:

```bash
# Open the help dialog of the CLI tool.
$ uv run python -m qnrl.cli --help
```

## Training 

> [!IMPORTANT]
> For all training you should create a `.env` file with key `WANDB_API_KEY`.

### Experiments

The experiment configuration of each of the frameworks and environments discussed in the paper is described as a YAML file in the [`./sweeps`](./sweeps) folder.

The full list of experiments is as follows:

Experiment YAML File | Description
--- | ---
[`./sweeps/c51_Acrobot-v1.yaml`](./sweeps/c51_Acrobot-v1.yaml) | C51 algorithm on the Acrobot environment.
[`./sweeps/qnrl_Acrobot-v1.yaml`](./sweeps/qnrl_Acrobot-v1.yaml) | QnRL algorithm on the Acrobot environment.
[`./sweeps/c51_CartPole-v1.yaml`](./sweeps/c51_CartPole-v1.yaml) | C51 algorithm on the CartPole environment.
[`./sweeps/qnrl_CartPole-v1.yaml`](./sweeps/qnrl_CartPole-v1.yaml) | QnRL algorithm on the CartPole environment.
[`./sweeps/c51_FrozenLake-v1.yaml`](./sweeps/c51_FrozenLake-v1.yaml) | C51 algorithm on the FrozenLake environment.
[`./sweeps/qnrl_FrozenLake-v1.yaml`](./sweeps/qnrl_FrozenLake-v1.yaml) | QnRL algorithm on the FrozenLake environment.
[`./sweeps/c51_CliffWalking-v1.yaml`](./sweeps/c51_CliffWalking-v1.yaml) | C51 algorithm on the CliffWalking environment.
[`./sweeps/qnrl_CliffWalking-v1.yaml`](./sweeps/qnrl_CliffWalking-v1.yaml) | QnRL algorithm on the CliffWalking environment.
[`./sweeps/c51_Breakout-v5.yaml`](./sweeps/c51_Breakout-v5.yaml) | C51 algorithm on the Breakout environment.
[`./sweeps/qnrl_Breakout-v5.yaml`](./sweeps/qnrl_Breakout-v5.yaml) | QnRL algorithm on the Breakout environment.
[`./sweeps/c51_SpaceInvaders-v5.yaml`](./sweeps/c51_SpaceInvaders-v5.yaml) | C51 algorithm on the SpaceInvaders environment.
[`./sweeps/qnrl_SpaceInvaders-v5.yaml`](./sweeps/qnrl_SpaceInvaders-v5.yaml) | QnRL algorithm on the SpaceInvaders environment.

### Sweeps using WandB

WandB makes it very easy to run hyperparameter sweeps to compare the performance of multiple configurations. The following sections detail how you can run our preconfigured sweeps using the `wandb` API.

#### Local computer

On your personal computer, start a new sweep:

```bash
# Start a sweep using YAML config via UV.
$ uv run --env-file .env wandb sweep sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep from: sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep with ID: 4ykq4hvw
wandb: View sweep at: https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/4ykq4hvw
wandb: Run sweep agent with: wandb agent zanderman-virginia-tech/qnrl/4ykq4hvw
# Copy the agent ID string: "zanderman-virginia-tech/qnrl/4ykq4hvw"
# Paste the agent ID string into the following command:
# Start 1 agent that performs 1 run of the sweep.
$ uv run --env-file .env wandb agent --count 1 zanderman-virginia-tech/qnrl/4ykq4hvw
```

#### SLURM

Source `.env`:

```bash
# Load Anaconda from SLURM modules.
$ module load Miniconda3
# Load Python environment from Anaconda.
$ conda activate qnrl
# Source environment file, which should have WandB API key.
$ source .env
# Start a sweep using YAML config file and pass WandB API key from environment.
$ WANDB_API_KEY=$WANDB_API_KEY wandb sweep sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep from: sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep with ID: 4ykq4hvw
wandb: View sweep at: https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/4ykq4hvw
wandb: Run sweep agent with: wandb agent zanderman-virginia-tech/qnrl/4ykq4hvw
# Copy the agent ID string: "zanderman-virginia-tech/qnrl/4ykq4hvw"
# Paste the agent ID string into the following command:
# Start 100 agents, max of 10 running at a time, that each perform 1 run of the sweep.
$ sbatch --array=1-100%10 scripts/slurm-wandb.sh agent --count 1 zanderman-virginia-tech/qnrl/4ykq4hvw
```

### Single-run Training

You can also train single hyperparameter configuration instances using our CLI via the command `python -m qnrl.cli train`. This commands takes a `--config="{...}"` argument that is **a Python dictionary formatted as a string**. The training CLI requires this type of configuration formatting to be seamlessly compatible with the WandB sweep API. As such, these are the same training commands that a WandB agent runs under the hood.

```bash
# Train a configuration of QnRL for the CartPole environment.
$ uv run --env-file .env python -m qnrl.cli train --config="{'batch_size': 128, 'buffer_size': 10000, 'env': {'env_id': 'CartPole-v1', 'type': 'EnvConfig'}, 'eps_end': 0.05, 'eps_exploration': 0.5, 'eps_start': 1, 'gamma': 0.99, 'learning_starts': 10000, 'model': {'din': 4, 'imprimitive': 'offset-CZ', 'input_multiplex_activation': 'tanh', 'input_preprocess': {'scale': [2.4, 2.5, 0.21, 2.5], 'type': 'RescaleInput'}, 'm_moment': 1, 'n_actions': 2, 'n_layers': 7, 'n_returns': 32, 'n_reuploads': 1, 'pauli_enc': ['RX'], 'pauli_var': ['RZ', 'RY', 'RZ'], 'rngs_kwargs': {'default': 0}, 'type': 'QuantumNativeQNetwork', 'z_max': 100, 'z_min': -100}, 'num_atoms': 32, 'optimizer': {'pytree': {'circ_weights_enc_obs_kernel': 'trainable_quantum', 'circ_weights_var': 'trainable_quantum', 'linear_input_projection': {'bias': 'trainable_classical', 'kernel': 'trainable_classical'}}, 'transforms': {'trainable_classical': {'b1': 0.7, 'b2': 0.999, 'learning_rate': 0.001, 'type': 'AdamW', 'weight_decay': 0.005}, 'trainable_quantum': {'b1': 0.7, 'b2': 0.999, 'learning_rate': 0.01, 'type': 'AdamW', 'weight_decay': 0.005}}, 'type': 'Partition'}, 'policy_update_frequency': 10, 'reseed_model_on_init': True, 'seed': 21, 'target_update_frequency': 500, 'tau': 1, 'total_timesteps': 100000, 'type': 'QnRL'}" --eval_seed=1024 --num_eval=10
```

## Authors

- [zanderman](https://github.com/zanderman) [![zanderman github](https://img.shields.io/badge/GitHub-zanderman-181717.svg?style=flat&logo=github)](https://github.com/zanderman)
- [saadwalid](https://github.com/saadwalid) [![saadwalid github](https://img.shields.io/badge/GitHub-saadwalid-181717.svg?style=flat&logo=github)](https://github.com/saadwalid)