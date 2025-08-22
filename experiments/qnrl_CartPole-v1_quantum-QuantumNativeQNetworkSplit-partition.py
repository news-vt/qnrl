from pathlib import Path

from qnrl.algorithms.qnrl import *
from qnrl.optimizer import *
from qnrl.metric import *
from qnrl.modules import *
from qnrl.gym import *
from qnrl.checkpoint import *

FILEPATH = Path(__file__)

# Unique experiment name. This will be used to identify metrics files and checkpoints.
NAME = FILEPATH.stem # Same as filename without extension.

# CartPole-v1
din = 4
n_actions = 2

n_returns = 101
z_min = -100
z_max = 100
m_moment = 1
n_reuploads = 2
n_layers = 5
# diff_method="backprop"
# shots=None
diff_method=None
shots=100

# Experiment configuration.
config = QnRL(
    metric_manager=MetricManager(
        path=(FILEPATH.parent/"../metrics"/Path(f"{NAME}.json")).resolve(),
        save_frequency=100, # Every 100 steps.
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir=(FILEPATH.parent/"../checkpoints"/Path(f"{NAME}")).resolve(),
        checkpoint_options=CheckpointOptions(
            save_interval_steps=1,
            max_to_keep=1,
            enable_async_checkpointing=True,
        ),
    ),
    env=ScaleObservation(
        env=EnvConfig(env_id="CartPole-v1"),
        scale=[2.4, 2.5, 0.21, 2.5],
    ),
    model=QuantumNativeQNetworkSplit(
        din=din,
        n_actions=n_actions,
        n_returns=n_returns,
        z_min=z_min,
        z_max=z_max,
        m_moment=m_moment,
        n_reuploads=n_reuploads,
        n_layers=n_layers,
        diff_method=diff_method,
        shots=shots,
        rngs_kwargs=dict(default=0),
    ),
    optimizer=Partition(
        transforms={
            'key_circuit_weights': Adam(learning_rate=1.0e-2),
            'key_layer_project_input': Adam(learning_rate=1.0e-1),
        },
        pytree={
            'circ_weights': 'key_circuit_weights',
            'layer_project_input': {
                'bias': 'key_layer_project_input',
                'kernel': 'key_layer_project_input',
            },
        },
    ),
    policy_type='stochastic',
)