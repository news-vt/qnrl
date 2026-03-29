from dataclasses import dataclass
from flax import nnx
import jax
import jax.numpy as jnp
from . import QuantumNativeQNetwork



@dataclass(kw_only=True)
class QuantumNativeQNetworkConv(QuantumNativeQNetwork):
    """Quantum-native (distributional) Q-value network for 3D image data of shape `(height, width, feature)` using convolutional preprocessing layers."""
    w: int # Image width.
    h: int # Image height.
    frame_stack: int # Number of frames that will be stacked together.
    conv_douts: list[int] # List of convolutional output dimensions.
    conv_kernel_size: list[tuple[int,int]] # Msut be same length as `conv_douts`.
    conv_strides: list[tuple[int,int]] # Msut be same length as `conv_douts`.


    @staticmethod
    def compute_conv_output_shape(l_in, padding, dilation, kernel_size, stride):
        """Compute image dimension after applying padding, dilation, kernel size, and stride.
        
        See: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        return int(jnp.floor((l_in + 2.*padding - dilation * (kernel_size - 1) - 1)/stride + 1))

    def pre_reseed(self, rngs_kwargs: dict[str, int]):
        
        # Call parent method to perform pre-reseed operations.
        # This is necessary to do before the operations below so that we can overwrite some parent variables for the reseed operation.
        super().pre_reseed(rngs_kwargs=rngs_kwargs)
        
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
        # Call reseed on parent to create seeded parameters.
        super().post_reseed(rngs_kwargs=rngs_kwargs)
        
        # Convolutions.
        dims = [self.din*self.frame_stack, *self.conv_douts]
        din_dout = list(zip(dims[:-1], dims[1:])) # (din, dout)
        self.convs = {
            f"convs_{i}": nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=self.conv_kernel_size[i], strides=self.conv_strides[i], padding="VALID", rngs=self.rngs)
            for i, (in_features, out_features) in enumerate(din_dout)
        }

    @nnx.jit
    def pre_project_observations(self, x: jax.Array): # (batch,din,)
        x = super().pre_project_observations(x)
        ###
        # Convert to shape: (batch, w, h, din, frame_stack)
        x = jnp.asarray(x).reshape((-1, self.frame_stack, self.w, self.h, self.din,))
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Combine `frame_stack` and `din` into one feature dimension.
        x = x.reshape(*x.shape[:-2], -1)
        
        # Apply Conv+ReLU to all convolutions.
        for key, layer in self.convs.items():
            x = layer(x)
            x = nnx.relu(x)
        
        # Reshape to 2D (batch, feature).
        x = x.reshape((x.shape[0], -1))
        ###
        return x