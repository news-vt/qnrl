from pathlib import Path

from qnrl.algorithms.c51 import *
from qnrl.optimizer import *
from qnrl.metric import *
from qnrl.modules import *
from qnrl.gym import *
from qnrl.checkpoint import *

FILEPATH = Path(__file__)

# Unique experiment name. This will be used to identify metrics files and checkpoints.
NAME = FILEPATH.stem # Same as filename without extension.

num_atoms = 101

# Experiment configuration.
config = C51(
    buffer_size=10_000,
    seed=0,
    batch_size=128,
    total_timesteps=500_000,
    learning_starts=10_000,
    policy_update_frequency=10,
    target_update_frequency=500,
    gamma=0.99,
    tau=1.0,
    eps_start=1,
    eps_end=5e-2,
    eps_exploration=0.5,
    #
    metric_manager=MetricManager(
        metric_dir=(FILEPATH.parent/"../metrics"/Path(f"{NAME}")).resolve(),
        metric_file_type="json",
        save_interval_steps=100,
        max_to_keep=2,
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir=(FILEPATH.parent/"../checkpoints"/Path(f"{NAME}")).resolve(),
        checkpoint_options=CheckpointOptions(
            save_interval_steps=1,
            max_to_keep=1,
            enable_async_checkpointing=True,
        ),
    ),
    env=EnvConfig(env_id="MountainCar-v0"),
    num_atoms=num_atoms,
    z_min=-100,
    z_max=100,
    model=Sequential(
        layers={
            'QNetworkLinear': DistQNetworkLinear(
                din=2,
                num_actions=3,
                num_atoms=num_atoms,
                douts=[120, 84],
                rngs_kwargs=dict(default=42),
            ),
        },
    ),
    optimizer=Adam(learning_rate=2.5e-4),
)