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
    metric_manager=MetricManager(
        path=(FILEPATH.parent/"../metrics"/Path(f"{NAME}.json")).resolve(),
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir=(FILEPATH.parent/"../checkpoints"/Path(f"{NAME}")).resolve(),
        checkpoint_options=CheckpointOptions(
            save_interval_steps=1,
            max_to_keep=1,
            enable_async_checkpointing=True,
        ),
    ),
    gym=GymConfig(env_id="CartPole-v1"),
    num_atoms=num_atoms,
    z_min=-100,
    z_max=100,
    model_policy=Sequential(
        layers={
            'RescaleInput': RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]),
            'QNetworkLinear': DistQNetworkLinear(
                din=4,
                num_actions=2,
                num_atoms=num_atoms,
                douts=[120, 84],
                rngs_kwargs=dict(default=42),
            ),
        },
    ),
    model_target=Sequential(
        layers={
            'RescaleInput': RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]),
            'QNetworkLinear': DistQNetworkLinear(
                din=4,
                num_actions=2,
                num_atoms=num_atoms,
                douts=[120, 84],
                rngs_kwargs=dict(default=42),
            ),
        },
    ),
    optimizer=Adam(learning_rate=2.5e-4),
)