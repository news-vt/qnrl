# QnRL: Quantum-native Reinforcement Learning

## Training sweeps using WandB

For all training you should create a `.env` file with key `WANDB_API_KEY`.

### Local computer

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

### SLURM

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

## Authors

- [zanderman](https://github.com/zanderman) [![zanderman github](https://img.shields.io/badge/GitHub-zanderman-181717.svg?style=flat&logo=github)](https://github.com/zanderman)
- [saadwalid](https://github.com/saadwalid) [![saadwalid github](https://img.shields.io/badge/GitHub-saadwalid-181717.svg?style=flat&logo=github)](https://github.com/saadwalid)