from dataclasses import dataclass
from functools import reduce

import jax
from flax import nnx
import jax.numpy as jnp

from .serializable import SerializableModule


from .quantumdistqnetworklinear import batched_run_circuit # Use circuit from linear network.


@dataclass(kw_only=True)
class QuantumDistQNetworkConv(SerializableModule):
    """Quantum Distributional Q-learning model using convolutional input and linear (i.e., fully connected) output projection layers."""
    din: int
    num_actions: int
    num_atoms: int
    num_wires: int
    num_layers: int
    input_preprocess: SerializableModule | None = None
    """Nested module for input preprocessing (can be trainable)."""
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = "backprop"
    shots: int | None = None
    ####
    w: int # Image width.
    h: int # Image height.
    frame_stack: int # Number of frames that will be stacked together.
    conv_douts: list[int] # List of convolutional output dimensions.
    conv_kernel_size: list[tuple[int,int]] # Must be same length as `conv_douts`.
    conv_strides: list[tuple[int,int]] # Must be same length as `conv_douts`.
    
    @staticmethod
    def compute_conv_output_shape(l_in, padding, dilation, kernel_size, stride):
        """Compute image dimension after applying padding, dilation, kernel size, and stride.
        
        See: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        return int(jnp.floor((l_in + 2.*padding - dilation * (kernel_size - 1) - 1)/stride + 1))

    def __post_init__(self):
        
        # Create parameters that depend on random seed.
        self.reseed(self.rngs_kwargs)
    
    def pre_reseed(self, rngs_kwargs: dict[str, int]):
        # self.circ_observation_shape_in = self.din # Dimension of the input data.
        self.circ_observation_shape_out = (self.num_layers,self.num_wires,1,) # Each qubit has (Rx,).
        self.circ_weights_shape = (self.num_layers+1,self.num_wires,3,) # Each qubit has (Rz, Ry, Rz).
        self.circ_outputs_shape_in = self.num_wires
        self.circ_outputs_shape_out = (self.num_actions, self.num_atoms,)
        
        # Ensure convolution parameters are valid.
        assert len(self.conv_douts) == len(self.conv_kernel_size) == len(self.conv_strides)
        
        # Compute image dimensions after each convolution.
        input_dilation = 1
        w, h = self.w, self.h
        for i in range(len(self.conv_kernel_size)):
            h = self.compute_conv_output_shape(h, padding=0, dilation=input_dilation, kernel_size=self.conv_kernel_size[i][0], stride=self.conv_strides[i][0])
            w = self.compute_conv_output_shape(w, padding=0, dilation=input_dilation, kernel_size=self.conv_kernel_size[i][1], stride=self.conv_strides[i][1])
        # Input data to quantum circuit will be final convolution dim * image height * image width.
        # We purposefully overwrite this variable from the parent here.
        self.circ_observation_shape_in = self.conv_douts[-1] * h * w

    def post_reseed(self, rngs_kwargs: dict[str, int]):
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        
        
        # Convolutions.
        dims = [self.din*self.frame_stack, *self.conv_douts]
        din_dout = list(zip(dims[:-1], dims[1:])) # (din, dout)
        self.convs = {
            f"convs_{i}": nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=self.conv_kernel_size[i], strides=self.conv_strides[i], padding="VALID", rngs=self.rngs)
            for i, (in_features, out_features) in enumerate(din_dout)
        }
        
        
        # Project shape of final classical layer to shape compatible with quantum circuit input.
        self.layer_project_input = nnx.Linear(in_features=self.circ_observation_shape_in, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs)
        
        # Project shape of quantum circuit output to desired model output shape.
        self.layer_project_output = nnx.Linear(in_features=self.circ_outputs_shape_in, out_features=reduce(lambda x,y: x*y, self.circ_outputs_shape_out), rngs=self.rngs)

    def reseed(self, rngs_kwargs: dict[str, int]):
        """Re-create model parameters using a dictionary of `nnx.Rngs` keyword arguments."""
        self.pre_reseed(rngs_kwargs=rngs_kwargs)
        self.rngs_kwargs = rngs_kwargs
        self.rngs: nnx.Rngs = nnx.Rngs(**self.rngs_kwargs)
        self.post_reseed(rngs_kwargs=rngs_kwargs)
    
    @nnx.jit
    def __call__(self, x):
        
        # Preprocess input data.
        if self.input_preprocess is not None:
            x = self.input_preprocess(x)
        
        # Convert to shape: (batch, w, h, din, frame_stack)
        x = jnp.asarray(x).reshape((-1, self.frame_stack, self.w, self.h, self.din,))
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Combine `frame_stack` and `din` into one feature dimension.
        x = x.reshape(*x.shape[:-2], -1)
        
        # Apply Conv+ReLU to all convolutions.
        for key, layer in self.convs.items():
            x = layer(x)
            x = nnx.relu(x)
        
        # Reshape to 2D (batch, feature).
        x = x.reshape((x.shape[0], -1))
        
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_observation_shape_out,))
        
        # Circuit input activation.
        x = jnp.tanh(x)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Output of circuit will be reverse-shaped list with shape `(num_wires,batch,)` due to how batching is handled with pennylane measurements.
        x = batched_run_circuit(
            key=subkeys,
            device_name=self.device_name,
            diff_method=self.diff_method,
            shots=self.shots,
            inputs=x,
            weights=self.circ_weights,
            num_wires=self.num_wires,
            num_layers=self.num_layers,
        )
        # # Reverse the shape to get back into a form acceptable for future layers.
        x = jnp.array(x).T # (batch,num_wires,)
        x = self.layer_project_output(x) # (batch,n_actions*n_atoms,)
        x = x.reshape((-1, *self.circ_outputs_shape_out)) # (batch, n_actions, n_atoms,)
        x = nnx.softmax(x, axis=-1) # Distribution over the atoms for every action.
        return x