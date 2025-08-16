from pathlib import Path

from qnrl.algorithms.dqn import *
from qnrl.optimizer import *
from qnrl.metric import *
from qnrl.modules import *
from qnrl.gym import *

FILEPATH = Path(__file__)

# Unique experiment name. This will be used to identify metrics files and checkpoints.
NAME = FILEPATH.stem # Same as filename without extension.

# Experiment configuration.
config = DQN(
    metric_manager=MetricManager(
        path=(FILEPATH.parent/"../metrics"/Path(f"{NAME}.json")).resolve(),
    ),
    gym=GymConfig(env_id="CartPole-v1"),
    model_policy=Sequential(
        layers={
            'RescaleInput': RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]),
            'QuantumQNetworkLinear': QuantumQNetworkLinear(
                din=4,
                num_actions=2,
                num_wires=10,
                num_layers=5,
                rngs_kwargs=dict(default=42),
                diff_method="backprop",
                shots=None,
            ),
        },
    ),
    model_target=Sequential(
        layers={
            'RescaleInput': RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]),
            'QuantumQNetworkLinear': QuantumQNetworkLinear(
                din=4,
                num_actions=2,
                num_wires=10,
                num_layers=5,
                rngs_kwargs=dict(default=43),
                diff_method="backprop",
                shots=None,
            ),
        },
    ),
    optimizer=Partition(
        transforms={
            'key_circuit_weights': Adam(learning_rate=1.0e-2),
            'key_layer_project_input': Adam(learning_rate=1.0e-1),
            'key_layer_project_output': Adam(learning_rate=1.0e-1),
        },
        pytree={
            'layers': {
                'QuantumQNetworkLinear': {
                    'circ_weights': 'key_circuit_weights',
                    'layer_project_input': {
                        'bias': 'key_layer_project_input',
                        'kernel': 'key_layer_project_input',
                    },
                    'layer_project_output': {
                        'bias': 'key_layer_project_output',
                        'kernel': 'key_layer_project_output',
                    },
                },
            },
        },
    ),
)