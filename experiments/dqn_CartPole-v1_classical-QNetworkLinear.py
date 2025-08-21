from pathlib import Path

from qnrl.algorithms.dqn import *
from qnrl.optimizer import *
from qnrl.metric import *
from qnrl.modules import *
from qnrl.gym import *
from qnrl.checkpoint import *

FILEPATH = Path(__file__)

# Unique experiment name. This will be used to identify metrics files and checkpoints.
NAME = FILEPATH.stem # Same as filename without extension.

# Experiment configuration.
config = DQN(
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
    env=EnvConfig(env_id="CartPole-v1"),
    model_policy=Sequential(
        layers={
            'RescaleInput': RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]),
            'QNetworkLinear': QNetworkLinear(
                din=4,
                num_actions=2,
                douts=[120, 84],
                rngs_kwargs=dict(default=42),
            ),
        },
    ),
    model_target=Sequential(
        layers={
            'RescaleInput': RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]),
            'QNetworkLinear': QNetworkLinear(
                din=4,
                num_actions=2,
                douts=[120, 84],
                rngs_kwargs=dict(default=43),
            ),
        },
    ),
    optimizer=Adam(learning_rate=2.5e-4),
)