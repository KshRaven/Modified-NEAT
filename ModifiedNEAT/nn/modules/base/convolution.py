from ...base import NeatModule, NeatParameter
from .util import calc_padding

from torch import Tensor
from typing import Union, Iterable

import torch
import torch.nn.functional as F


class Convolution(NeatModule):
    """Base convolution layer supporting n-dimensional convolutions."""
    
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, Iterable[int]], stride=1,
                 dilation=1, padding: Union[int, Iterable[int]] = 0, padding_mode='zeros', groups=1,
                 ndim=1, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        # ====================== Input Handling ======================
        if channels_in % groups != 0:
            raise ValueError(f"Input channels ({channels_in}) must be divisible by groups ({groups})")
        if channels_out % groups != 0:
            raise ValueError(f"Output channels ({channels_out}) must be divisible by groups ({groups})")
        if isinstance(kernel_size, (int, float)):
            kernel_size = tuple([int(kernel_size) for _ in range(ndim)])
        elif len(kernel_size) != ndim:
            raise ValueError(f"Kernel dimensions '{kernel_size}' do not match convolution dimensions '{ndim}'")
        if not isinstance(kernel_size, tuple):
            kernel_size = tuple(kernel_size)
        padding_value = 0
        if padding is not None:
            if isinstance(padding, (int, float)):
                padding = [int(padding) for _ in range(ndim)]
            elif len(kernel_size) != ndim:
                raise ValueError(f"Padding dimensions '{padding}' do not match convolution dimensions '{ndim}'")
            if not isinstance(padding, list):
                padding = list(padding)
            if any([pv < -1 or not isinstance(pv, int) for pv in padding]):
                raise ValueError(f"Invalid {self.__class__.__name__} padding '{padding}'")
            for pi, pv in enumerate(padding):
                if pv == -1:
                    padding[pi] = calc_padding(kernel_size[pi], stride, dilation)
            padding = tuple(padding)
            self._mean_padding = True if padding_mode == 'mean' else False
            if padding_mode in ['zeros', 'mean']:
                padding_mode = 'constant'
            elif isinstance(padding_mode, (int, float)):
                padding_value = padding_mode
                padding_mode = 'constant'
            elif padding_mode not in ['constant', 'reflect', 'replicate', 'circular']:
                padding_value = None
                raise ValueError(f"Supported padding modes are ['zeros', 'constant', <constant numerical value>, "
                                 f"'reflect', 'replicate', 'circular', 'mean']")
            if self._mean_padding:
                raise NotImplementedError(f"Mean padding is not avaialble yet!")
        # print(kernel_size, padding)
        assert len(kernel_size) == len(padding)

        # TODO: Fix how modules are displayed
        super(Convolution, self).__init__(
            channels_in=channels_in, channels_out=channels_out, kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias if bias is True else False,
            padding_mode='zeros' if padding_mode == 'constant' and padding_value == 0 else padding_mode
        )

        # Attributes
        self.channels_in    = channels_in
        self.channels_out   = channels_out
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.dilation       = dilation
        self.groups         = groups
        self.bias           = bias
        self.padding        = padding
        self.padding_mode   = padding_mode
        self.padding_value  = padding_value
        self.ndim           = ndim

        # Parameters
        self.kernels = NeatParameter(shape=(channels_out, channels_in // groups, *kernel_size),
                                     requires_grad=False, device=device, dtype=dtype)
        self.biases = NeatParameter(shape=(channels_out,), requires_grad=False, device=device, dtype=dtype) \
            if bias else None

        # States
        self.device = device
        self.dtype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        verbose = verbose is not None and verbose >= 3
        if verbose:
            print(f"Inputs = {tensor.shape}")
        # Expected shape = (genomes, batch_size, channels_in, seq_len)
        conv_dims = len(self.kernel_size)
        extra_shape, channels_in, input_seq_len = \
            tensor.shape[:-(conv_dims+1)], tensor.shape[-(conv_dims+1)], tensor.shape[-conv_dims:]
        # kernels = self.expand(self.kernels[keys], tensor, keys=keys)
        kernels = self.kernels[keys]
        if verbose:
            print(f"Kernel = {kernels.shape}")
            print(f"extra_shape = {extra_shape}")
            print(f"channels_in = {channels_in}")
            print(f"input_seq_len = {input_seq_len}")

        # ====================== Group Validation ======================
        if channels_in % self.groups != 0:
            raise ValueError(f"Input channels ({channels_in}) must be divisible by groups ({self.groups})")
        if (channels_in // self.groups) != kernels.shape[-(conv_dims+1)]:
            raise ValueError(f"Kernel input channels ({kernels.shape[-(conv_dims+1)]}) must equal to "
                             f"channels_in//groups ({channels_in}//{self.groups} = {channels_in//self.groups})")

        # ====================== Padding Handling ======================
        if any([pv != 0 for pv in self.padding]):
            padding = tuple([])
            for pv in self.padding:
                padding += (pv, pv)
            # if self.padding_mode != 'constant' and self.ndim < 3:
            #     # for _ in range(max(0, self.ndim-3)):
            #     padding += (0, 0)
            if verbose:
                print(f"padding = {padding}")
            tensor = tensor.view(-1, channels_in, *input_seq_len)
            # print(tensor.shape)
            tensor = F.pad(
                tensor, padding, mode=self.padding_mode, value=self.padding_value,
            )
            padded_seq_len = tensor.shape[2:]
            tensor = tensor.contiguous().view(*extra_shape, channels_in, *padded_seq_len)

        padded_seq_len: tuple[int, ...] = tensor.shape[-len(self.kernel_size):]
        if verbose:
            print(f"Padded input: {tensor.shape}")

        # ====================== Kernel/Window Calculations ======================
        eff_kernel_len = tuple([(kernel_size - 1) * self.dilation + 1 for kernel_size in self.kernel_size])
        output_seq_len = tuple([(padded_seq_len[di] - ekl) // self.stride + 1 for di, ekl in enumerate(eff_kernel_len)])

        if verbose:
            print(f"Effective kernel length: {eff_kernel_len}")
            print(f"Calculated output size: {output_seq_len}")

        if any([ekl > psl for ekl, psl in zip(eff_kernel_len, padded_seq_len)]):
            raise ValueError(f"Effective kernel length {eff_kernel_len} exceeds padded input size {padded_seq_len}")
        if any([osl <= 0 for osl in output_seq_len]):
            raise ValueError(f"Invalid output size: {output_seq_len}")

        # ====================== Window Generation ======================
        windows = tensor
        # Extract windows [batch, ch_in, output_size, kernel_size]
        # TODO: Support dimensional stride and dilation
        for i, k_dim in enumerate(eff_kernel_len):
            # eff_kernel_size = (k_dim - 1) * self.dilation + 1
            windows = windows.unfold(len(extra_shape)+1+i, k_dim, self.stride)  # Unfold along spatial dimensions
            if self.dilation > 1:
                windows = windows[..., ::self.dilation]

        if verbose:
            print(f"Windows tensor shape: {windows.shape}")
            if verbose >= 2:
                print(f"First window sample:\n{windows[*[0 for _ in range(len(extra_shape)+1)]]}")

        # ====================== Group Processing ======================
        windows_grouped = windows.view(
            *extra_shape, self.groups, self.channels_in//self.groups, *output_seq_len, *self.kernel_size
        )
        # genomes = self.kernels.shape[0]
        kernel_grouped = kernels.view(
            -1, self.groups, self.channels_out//self.groups, self.channels_in//self.groups, *self.kernel_size
        )

        if verbose:
            print(f"Grouped windows shape: {windows_grouped.shape}")
            print(f"Grouped kernel shape: {kernel_grouped.shape}")
            if verbose >= 2:
                print(f"First group window sample:\n{windows_grouped[0,0,0,0]}")
                print(f"First group kernel sample:\n{kernel_grouped[0,0,0]}")

        # ====================== Grouped Convolution ======================
        w = "xyz"[:len(output_seq_len)]
        k = "abc"[:len(self.kernel_size)]
        # windows = (..., batch_size, groups, channels_in, *out_pixels_dims, *kernel_dims)
        # kernel = (...., groups, channels_out, channels_in, *kernel_dims)
        # output = (..., batch_size, groups, channels_out, *out_pixels_dims)
        try:
            tensor = torch.einsum(f"h...gi{w}{k},h...goi{k}->h...go{w}", windows_grouped, kernel_grouped)
        except Exception as e:
            if not verbose:
                self.forward(tensor, keys, verbose=2)
            print(f"windows = {windows_grouped.shape}")
            print(f"kernels = {kernel_grouped.shape}")
            raise e

        if verbose:
            print(f"Grouped output shape: {tensor.shape}")
            if verbose >= 2:
                print(f"First group output sample:\n{tensor[0,0,0]}")

        # Combine groups [batch, channels_out, output_size]
        tensor = tensor.contiguous().view(*extra_shape, -1, *output_seq_len)
        if self.biases is not None:
            # biases = self.expand(self.biases[keys], tensor, offset=1, keys=keys)
            biases = self.biases[keys]
            bias_shape = [extra_shape[0]] + [1 for _ in range(max(0, len(extra_shape)-1))] + [-1] + [1 for _ in range(len(input_seq_len))]
            biases = biases.view(*bias_shape)
            if verbose:
                print(f"Tensor = {tensor.shape}")
                print(f"Bias = {biases.shape}")
            tensor = tensor + biases

        if verbose:
            print(f"Final output shape: {tensor.shape}")
            if verbose >= 2:
                print(f"First output channel sample:\n{tensor[0,0]}")

        return tensor


class Conv1d(Convolution):
    """1D Convolution."""
    
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, tuple[int, int]], stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=torch.device('cpu'), dtype=torch.float32):
        super(Conv1d, self).__init__(
            channels_in, channels_out, kernel_size, stride, dilation, padding, padding_mode,
            groups, 1, bias, device, dtype
        )


class Conv2d(Convolution):
    """2D Convolution."""
    
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, tuple[int, int]], stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=torch.device('cpu'), dtype=torch.float32):
        super(Conv2d, self).__init__(
            channels_in, channels_out, kernel_size, stride, dilation, padding, padding_mode,
            groups, 2, bias, device, dtype
        )


class Conv3d(Convolution):
    """3D Convolution."""
    
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, tuple[int, int]], stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=torch.device('cpu'), dtype=torch.float32):
        super(Conv3d, self).__init__(
            channels_in, channels_out, kernel_size, stride, dilation, padding, padding_mode,
            groups, 3, bias, device, dtype
        )
