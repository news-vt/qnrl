from dataclasses import dataclass, field
from flax import nnx
import jax.numpy as jnp

from . import SerializableModule

@dataclass(kw_only=True)
class DistQNetworkConv(SerializableModule):
    """Distributional Q-learning network with convolutional layers and support for frame stacking."""
    w: int # Image width.
    h: int # Image height.
    frame_stack: int # Number of frames that will be stacked together.
    din: int # Number of image feature dimensions (e.g., `din=3` for RGB image, `din=1` for greyscale image).
    num_actions: int
    num_atoms: int
    conv_douts: list[int] # List of convolutional output dimensions.
    conv_kernel_size: list[tuple[int,int]] # Msut be same length as `conv_douts`.
    conv_strides: list[tuple[int,int]] # Msut be same length as `conv_douts`.
    linear_douts: list[int] = field(default_factory=list) # List of linear layers applied after convolutions and before distribution projection.
    rngs_kwargs: dict[str, int]
    
    
    def __post_init__(self):
        self.reseed(self.rngs_kwargs)
    
    @staticmethod
    def compute_conv_output_shape(l_in, padding, dilation, kernel_size, stride):
        return int(jnp.floor((l_in + 2.*padding - dilation * (kernel_size - 1) - 1)/stride + 1))
    
    def reseed(self, rngs_kwargs: dict[str, int]):
        """Re-create model parameters using a dictionary of `nnx.Rngs` keyword arguments."""
        self.rngs_kwargs = rngs_kwargs
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        
        
        # Convolutions.
        assert len(self.conv_douts) == len(self.conv_kernel_size) == len(self.conv_strides)
        dims = [self.din*self.frame_stack, *self.conv_douts]
        din_dout = list(zip(dims[:-1], dims[1:])) # (din, dout)
        self.convs = [
            nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=self.conv_kernel_size[i], strides=self.conv_strides[i], padding="VALID", rngs=rngs)
            for i, (in_features, out_features) in enumerate(din_dout)
        ]
        # Compute image dimensions after each convolution.
        w, h = self.w, self.h
        for i in range(len(self.convs)):
            h = self.compute_conv_output_shape(h, padding=0, dilation=self.convs[i].input_dilation, kernel_size=self.conv_kernel_size[i][0], stride=self.conv_strides[i][0])
            w = self.compute_conv_output_shape(w, padding=0, dilation=self.convs[i].input_dilation, kernel_size=self.conv_kernel_size[i][1], stride=self.conv_strides[i][1])
        # Preserve output dimension of final convolution.
        final_conv_out_features = self.convs[-1].out_features
        
        # Organize dimensions so that the output of one is the input of another.
        dims = [final_conv_out_features*h*w, *self.linear_douts, self.num_actions * self.num_atoms]
        din_dout = list(zip(dims[:-1], dims[1:])) # (din, dout)
        self.linears = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in din_dout
        ]
    
    @nnx.jit
    def __call__(self, x): # (batch, frame_stack, w, h, din)
        # Ensure input shape with batch dimension (defaults to `batch=1`).
        x = jnp.asarray(x)
        x = x.reshape((-1, self.frame_stack, self.w, self.h, self.din,))
        
        # Convert to shape: (batch, w, h, din, frame_stack)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Combine `frame_stack` and `din` into one feature dimension.
        x = x.reshape(*x.shape[:-2], -1)
        
        # Apply Conv+ReLU to all convolutions.
        for layer in self.convs:
            x = layer(x)
            x = nnx.relu(x)
        
        # Reshape to 2D (batch, feature).
        x = x.reshape((x.shape[0], -1))
        
        # Apply Linear+ReLU to all linears except last layer.
        for layer in self.linears[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        
        # Generate return-atom distribution using last layer.
        x = self.linears[-1](x)
        x = x.reshape((-1, self.num_actions, self.num_atoms)) # (batch, action, atom)
        x = nnx.softmax(x, axis=-1) # Distribution over the atoms for every action.
        return x