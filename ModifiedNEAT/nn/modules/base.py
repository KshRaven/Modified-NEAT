
from ModifiedNEAT.nn.base import NeatModule, NeatParameter, _addindent
from ModifiedNEAT.util.fancy_text import CM, Fore

from torch import Tensor
from typing import Union, Iterable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import inspect
import math


def calc_padding(kernel_size: int, stride=1, dilation=1):
    return math.ceil(((kernel_size - 1) * dilation - (stride - 1)) / 2)


def display(label: str, var, check=None):
    if not (var if check is None else check):
        return ''
    else:
        return f", {label}={var}"


def get_conv(inputs: Union[Iterable[int], int]):
    if isinstance(inputs, (int, float)):
        inputs = [1 for _ in range(int(inputs))]
    if len(inputs) == 1:
        return Conv1d
    elif len(inputs) == 2:
        return Conv2d
    elif len(inputs) == 3:
        return Conv3d
    else:
        raise ValueError(f"Unsupported num of image dimension '{len(inputs)}'")


def has_parameter(method, param_name: str):
    """Check if a method has a specific parameter."""
    sig = inspect.signature(method)
    return param_name in sig.parameters


class Sequential(NeatModule):
    def __init__(self, *modules: NeatModule):
        super(Sequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        for m_idx, module in enumerate(self.modules_list):
            try:
                if isinstance(module, NeatModule):
                    tensor = module(tensor, keys=keys)
                else:
                    tensor = module(tensor)
                # print(f"Module {m_idx} '{module.__class__.__name__}' = {tensor.shape}")
            except Exception as e:
                print(CM(f"Failed on module '{m_idx}' =>\n{module}\n"
                         f"\twith tensor shape {tensor.shape}", Fore.LIGHTRED_EX))
                raise e

        return tensor

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for index, module in enumerate(self.modules_list):
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + str(index) + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "[NeatModule]("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Linear(NeatModule):
    def __init__(self, inputs: int, outputs: int, bias=True,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(Linear, self).__init__(inputs=inputs, outputs=outputs, bias=bias)
        self.weights = NeatParameter((inputs, outputs), False, device, dtype)
        self.biases  = NeatParameter(outputs, False, device, dtype) if bias else None

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, verbose: int = None):
        try:
            # print(f"lin_mod={self.weights.data.shape, tensor.shape}")
            w = self.expand(self.weights[keys], tensor, keys=keys)
            # print(w.shape, tensor.shape)
            tensor = torch.matmul(tensor, w)
            if self.biases is not None:
                b = self.expand(self.biases[keys], tensor, keys=keys)
                tensor = tensor + b
            else:
                b = None

            if not verbose:
                return tensor
            else:
                return tensor, (w, b)
        except Exception as e:
            print(f"Input shape = {tensor.shape}")
            print(f"Weights shape = {self.weights.shape}")
            print(f"Biases shape = {self.biases.shape if self.biases is not None else None}")
            raise e


class LayerNorm(NeatModule):
    def __init__(self, normalized_shape: Union[int, Iterable[int]], eps=1e-8, elementwise_affine=True,
                 bias=True, device=torch.device('cpu'), dtype=torch.float32):
        if isinstance(normalized_shape, (numbers.Integral, numbers.Real)):
            normalized_shape = [int(normalized_shape)]
        normalized_shape = tuple(normalized_shape)
        super(LayerNorm, self).__init__(shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                                        bias=elementwise_affine and bias)
        self.normalized_shape = tuple(normalized_shape)
        self.dim = tuple([i for i in range(-len(self.normalized_shape), 0)])
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weights: NeatParameter = None
        self.biases: NeatParameter = None
        if self.elementwise_affine:
            self.weights = NeatParameter(normalized_shape, False, device, dtype)
            if bias:
                self.biases = NeatParameter(normalized_shape, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        mean        = torch.mean(tensor, dim=self.dim, keepdim=True)
        variance    = torch.var(tensor, dim=self.dim, keepdim=True, unbiased=False)
        tensor      = (tensor - mean) / torch.sqrt(variance + self.eps)

        if self.elementwise_affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, keys=keys)
            if self.biases is not None:
                tensor = tensor + self.expand(self.biases[keys], tensor, keys=keys)

        return tensor


class RMSNorm(NeatModule):
    def __init__(self, normalized_shape: Union[int, Iterable[int]], eps=1e-8,
                 elementwise_affine=True, device=torch.device('cpu'), dtype=torch.float32):
        if isinstance(normalized_shape, (numbers.Integral, numbers.Real)):
            normalized_shape = [int(normalized_shape)]
        normalized_shape = tuple(normalized_shape)
        super(RMSNorm, self).__init__(shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.normalized_shape = normalized_shape
        self.dim = tuple([i for i in range(-len(self.normalized_shape), 0)])
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weights: NeatParameter = None
        if elementwise_affine:
            self.weights = NeatParameter(normalized_shape, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        tensor = tensor / torch.sqrt(torch.mean(tensor**2, dim=self.dim, keepdim=True) + self.eps)

        if self.elementwise_affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, keys=keys)

        return tensor


class GroupNorm(NeatModule):
    def __init__(self, groups: int, channels: int, eps=1e-6, affine=True,
                 bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(GroupNorm, self).__init__(groups=groups, channels=channels, eps=eps, affine=affine, bias=affine and bias)
        if channels % groups != 0:
            raise ValueError("num_channels must be divisible by num_groups.")
        self.groups = groups
        self.channels = channels
        self.eps    = eps
        self.affine = affine
        self.bias   = bias

        self.weights: NeatParameter = None
        self.biases: NeatParameter  = None
        if self.affine:
            self.weights = NeatParameter(channels, False, device, dtype)
            if bias:
                self.biases = NeatParameter(channels, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None, permute: Iterable[int] = None):
        if permute is not None:
            assert len(permute) == tensor.ndim
            tensor = tensor.permute(permute)

        # Reshape the tensor to (N, G, C//G, *) for group normalization
        (genomes, batch_size, channels), pixels = tensor.shape[:3], tensor.shape[3:]
        if channels != self.channels:
            raise ValueError(f"Channels num do not match; Got '{channels}', expected '{self.channels}'")
        if self.weights is not None and (genomes > self.weights.shape[0] or genomes <= 0):
            raise ValueError(f"Genomes num do not match; Got '{genomes}', expected '{self.weights.shape[0]}'")
        tensor = tensor.view(genomes, batch_size, self.groups, channels//self.groups, *pixels)

        # Calculate mean and variance over the group and spatial dimensions
        dims = list(range(3, 4+len(pixels)))
        mean = tensor.mean(dim=dims, keepdim=True)
        var = tensor.var(dim=dims, keepdim=True, unbiased=False)

        # Normalize the tensor
        tensor = (tensor - mean) / torch.sqrt(var + self.eps)

        # Reshape back to (N, C, *)
        tensor = tensor.view(genomes, batch_size, channels, *pixels)

        if self.affine:
            # Apply affine transformation
            # print(f"tensor = {tensor.shape}")
            weights = self.expand(self.weights[keys], tensor, offset=1, keys=keys)
            # print(f"weight = {weights.shape}")
            tensor = tensor * weights
            if self.bias:
                tensor = tensor + self.expand(self.biases[keys], tensor, offset=1, keys=keys)

        if permute is not None:
            tensor = tensor.permute(permute)
        return tensor


class BatchNorm(NeatModule):
    def __init__(self, groups: int, channels: int, eps=1e-8, affine=True,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(BatchNorm, self).__init__(num_groups=groups, num_channels=channels, eps=eps, affine=affine)
        if channels % groups != 0:
            raise ValueError("num_channels must be divisible by num_groups.")
        self.num_groups = groups
        self.num_channels = channels
        self.dim = [1, 2]
        self.eps = eps
        self.affine = affine

        self.weights: NeatParameter = None
        self.biases: NeatParameter = None
        if affine:
            self.weights = NeatParameter(channels, False, device, dtype)
            self.biases = NeatParameter(channels, False, device, dtype)

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        # assuming shape (genomes, groups, channels, *extra)
        mean     = tensor.mean(dim=self.dim, keepdim=True)
        variance = tensor.var(dim=self.dim, unbiased=False, keepdim=True)
        tensor   = (tensor - mean) / torch.sqrt(variance + self.eps)

        if self.affine:
            tensor = tensor * self.expand(self.weights[keys], tensor, 1, keys=keys)
            tensor = tensor + self.expand(self.biases[keys], tensor, 1, keys=keys)
        return tensor


class Convolution(NeatModule):
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
            if padding_mode == 'zeros':
                padding_mode = 'constant'
            elif isinstance(padding_mode, (int, float)):
                padding_value = padding_mode
                padding_mode = 'constant'
            elif padding_mode not in ['constant', 'reflect', 'replicate', 'circular']:
                raise ValueError(f"Supported padding modes are ['zeros', 'constant', <constant numerical value>, "
                                 f"'reflect', 'replicate', 'circular']")
        # print(kernel_size, padding)
        assert len(kernel_size) == len(padding)

        super(Convolution, self).__init__(
            channels_in=channels_in, channels_out=channels_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode
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
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, tuple[int, int]], stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=torch.device('cpu'), dtype=torch.float32):
        super(Conv1d, self).__init__(channels_in, channels_out, kernel_size, stride, dilation, padding, padding_mode,
                                     groups, 1, bias, device, dtype)


class Conv2d(Convolution):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, tuple[int, int]], stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=torch.device('cpu'), dtype=torch.float32):
        super(Conv2d, self).__init__(channels_in, channels_out, kernel_size, stride, dilation, padding, padding_mode,
                                     groups, 2, bias, device, dtype)


class Conv3d(Convolution):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, tuple[int, int]], stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=torch.device('cpu'), dtype=torch.float32):
        super(Conv3d, self).__init__(channels_in, channels_out, kernel_size, stride, dilation, padding, padding_mode,
                                     groups, 3, bias, device, dtype)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, tensor: Tensor, keys: Union[int, Iterable[int]] = None):
        return tensor


class Transpose(nn.Module):
    def __init__(self, dim0=-1, dim1=-2):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, tensor: Tensor):
        return torch.transpose(tensor, self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f"dim0={self.dim0}, dim1={self.dim1}"


class Ignore(nn.Module):
    def __init__(self, rate: float = 0):
        super(Ignore, self).__init__()
        assert 0 <= rate < 1
        self.rate = rate

    def forward(self, tensor: Tensor):
        if self.training and torch.rand(1).item() < self.rate:
            tensor = tensor * 0
        return tensor

    def extra_repr(self) -> str:
        return f"rate={self.rate}"

