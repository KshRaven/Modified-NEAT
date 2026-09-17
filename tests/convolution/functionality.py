from ModifiedNEAT.nn.modules.base import calc_padding
from ModifiedNEAT.util.fancy_text import CM, Fore

import ModifiedNEAT as meat
import ModifiedNEAT.nn as mnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from torch import Tensor
from typing import Iterable, Union
from numba.core.errors import NumbaPerformanceWarning


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


class Permute(nn.Module):
    def __init__(self, permutation: list[int]):
        super(Permute, self).__init__()
        if not isinstance(permutation, Iterable):
            raise TypeError(f"Permutation must be iterable; received {type(permutation)}")
        self.permutation = permutation

    def forward(self, tensor: Tensor):
        return tensor.permute(self.permutation)


class Average(nn.Module):
    def __init__(self, dims=None, keepdims=False):
        super(Average, self).__init__()
        self.dims = dims
        self.keepdims = keepdims

    def forward(self, input: Tensor) -> Tensor:
        return torch.mean(input, dim=self.dims, keepdim=self.keepdims)


class AdaptiveAvgPooling(meat.NeatModule):
    def __init__(self, output_size: int | None | tuple[int | None, ...]):
        super(AdaptiveAvgPooling, self).__init__(output_size=output_size)
        self.output_size = output_size


class AdaptiveAvgPool1d(AdaptiveAvgPooling):
    def forward(self, tensor: Tensor):
        F.adaptive_avg_pool1d()
        pass


class Sequential(nn.Sequential):
    def forward(self, tensor: Tensor, verbose: int = None) -> Tensor:
        for module in self:
            if verbose:
                debugging = (f"Module '{CM(module.__class__.__name__, Fore.LIGHTMAGENTA_EX)}' input: "
                             f"{CM(tensor.shape, Fore.LIGHTCYAN_EX)}")
                if verbose >= 1:
                    pass
                if verbose >= 2:
                    pass
                print(debugging)
            tensor = module(tensor)
        return tensor


class NeatNetwork(meat.NeatModule):
    def __init__(self, inputs: int, outputs: int, dim_size: int, kernel_size: int, layers: int, groups=1,
                 epsilon=1e-12, bias=True, device='cpu', dtype=torch.float32):
        super(NeatNetwork, self).__init__()

        sequence = [
            mnn.Conv1d(inputs, dim_size, kernel_size, padding=-1, bias=bias, device=device, dtype=dtype),
            mnn.GroupNorm(groups, dim_size, epsilon, True, bias=bias, device=device, dtype=dtype),
            nn.SiLU(),
        ]
        for layer_idx in range(layers):
            sequence.extend([
                mnn.Conv1d(dim_size, dim_size, kernel_size, padding=-1, bias=bias, device=device, dtype=dtype),
                mnn.GroupNorm(groups, dim_size, epsilon, True, bias=bias, device=device, dtype=dtype),
                nn.SiLU(),
            ])
        sequence.extend([
            nn.Flatten(start_dim=-2),
            nn.AdaptiveAvgPool1d(dim_size),
            mnn.Linear(dim_size, dim_size, bias=bias, device=device, dtype=dtype),
            nn.SiLU(),
            mnn.Linear(dim_size, outputs, bias=bias, device=device, dtype=dtype),
        ])

        self.sequence = mnn.Sequential(*sequence)
        self.inputs = inputs
        self.outputs = outputs
        self.dim_size = dim_size
        self.kernel_size = kernel_size
        self.layers = layers
        self.groups = groups
        self.epsilon = epsilon
        self.bias = bias
        self.device = device
        self.dtype = dtype

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, us=True, verbose: int = None):
        # if us:
        #     tensor = tensor.unsqueeze(-2)
        tensor = self.sequence(tensor.contiguous(), keys=keys)
        return tensor

    def evaluate(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: int = None):
        logits = self.forward(tensor, keys=keys, verbose=verbose)
        return torch.argmax(logits, dim=-1)

    def guess(self, tensor, keys: Union[int, list[int]] = None):
        logits = self.forward(tensor, keys=keys)
        dist = torch.distributions.Categorical(F.softmax(logits, -1))
        return dist.sample()


class TorchNetwork(nn.Module):
    def __init__(self, genomes: int, inputs: int, outputs: int, dim_size: int, kernel_size: int, layers: int, groups=1,
                 epsilon=1e-12, bias=True, device='cpu', dtype=torch.float32):
        super(TorchNetwork, self).__init__()
        assert genomes > 0
        def cp(kernel_size: int):
            return calc_padding(kernel_size, 1, 1)

        self.genomes = nn.ModuleDict()
        for key, genome in enumerate(range(genomes)):
            sequence = [
                # Assuming dimension to convolve is last dimension and input is (batch_size, inp_features, seq_len)
                nn.Conv1d(inputs, dim_size, kernel_size, padding=cp(kernel_size), bias=bias, device=device, dtype=dtype),
                # Tensor here is (batch_size, dim_size, seq_len)
                nn.GroupNorm(groups, dim_size, epsilon, True, device=device, dtype=dtype),
                nn.SiLU(),
            ]
            for layer_idx in range(layers):
                sequence.extend([
                    nn.Conv1d(dim_size, dim_size, kernel_size, padding=cp(kernel_size), bias=bias, device=device, dtype=dtype),
                    nn.GroupNorm(groups, dim_size, epsilon, True, device=device, dtype=dtype),
                    nn.SiLU(),
                ])
            sequence.extend([
                # Tensor here is (batch_size, dim_size, seq_len)
                nn.Flatten(start_dim=-2),
                nn.AdaptiveAvgPool1d(dim_size),
                # Tensor here is (batch_size, dim_size)
                nn.Linear(dim_size, dim_size, bias=bias, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(dim_size, outputs, bias=bias, device=device, dtype=dtype),
                # Tensor here is (batch_size, out_features)
            ])
            self.genomes[str(key)] = Sequential(*sequence)

        self.inputs = inputs
        self.outputs = outputs
        self.dim_size = dim_size
        self.kernel_size = kernel_size
        self.layers = layers
        self.groups = groups
        self.epsilon = epsilon
        self.bias = bias
        self.device = device
        self.dtype = dtype

    def load_weights(self, neat_module: NeatNetwork):
        if not neat_module.genome_num == len(self.genomes):
            raise ValueError(f"Number of genomes in neat module ({neat_module.genome_num}) "
                             f"does not match number of genomes in torch module ({len(self.genomes)})")
        # if not len(list(neat_module.modules())) == len(list(self.modules())):
        #     raise ValueError(f"Number of modules in neat module ({len(list(neat_module.modules()))}) "
        #                      f"does not match number of modules in torch module ({len(list(self.modules()))})")
        modules: dict[int, dict[int, nn.Module]] = {}
        for key, genome in self.genomes.items():
            for idx, module in enumerate(genome.modules()):
                if idx not in modules:
                    modules[idx] = {}
                genome_idx = int(key)
                modules[idx][genome_idx] = module

        # Removing any NeatParameters
        neat_list = list(neat_module.modules())
        check_idx = 0
        while check_idx < len(neat_list):
            if isinstance(neat_list[check_idx], (meat.NeatParameter, nn.ModuleList, nn.ModuleDict, NeatNetwork)):
                neat_list.pop(check_idx)
            else:
                check_idx += 1

        # Debugging
        print(f"----- Neat Modules -----")
        for module in neat_list:
            print(module.__class__.__name__)
        print(f"----- Torch Modules -----")
        for module in self.genomes["0"].modules():
            print(module.__class__.__name__)
                
        skipped_list = []
        for source, destination in zip(neat_list, list(modules.values())):
            if isinstance(source, meat.NeatModule):
                for key in source.mapping:
                    genome_idx = source.mapping[key]
                    # if not hasattr(source, "biases"):
                    #     setattr(source, "biases", None)
                    if isinstance(source, mnn.Linear):
                        module: nn.Linear = destination[genome_idx]
                        module.weight = nn.Parameter(torch.transpose(source.weights.data[genome_idx], -1, -2))
                        if source.biases is not None:
                            if module.bias is not None:
                                module.bias = nn.Parameter(source.biases.data[genome_idx])
                            else:
                                raise ValueError(f"Missing bias parameter from source!")
                        else:
                            module.bias = None
                    elif isinstance(source, mnn.Conv1d):
                        module: nn.Conv1d = destination[genome_idx]
                        module.weight = nn.Parameter(source.kernels.data[genome_idx])
                        if source.biases is not None:
                            if module.bias is not None:
                                module.bias = nn.Parameter(source.biases.data[genome_idx])
                            else:
                                raise ValueError(f"Missing bias parameter from source!")
                        else:
                            module.bias = None
                    elif isinstance(source, mnn.GroupNorm):
                        module: nn.GroupNorm = destination[genome_idx]
                        module.weight = nn.Parameter(source.weights.data[genome_idx])
                        if not hasattr(module, "bias"):
                            setattr(module, "bias", None)
                        if source.bias is not None:
                            module.bias = nn.Parameter(source.biases.data[genome_idx])
                    else:
                        if isinstance(source, meat.NeatModule):
                            module_name = source.__class__.__name__
                            if module_name not in skipped_list:
                                print(f"Skipped NeatModule {CM(module_name, Fore.LIGHTRED_EX)}")
                                skipped_list.append(module_name)
                        else:
                            raise ValueError(f"Unsupported module type: {source.__class__.__name__}")

        assert not any([isinstance(m, meat.NeatModule) for m in self.modules()])

    def forward(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: int = None):
        # Expected shape (genomes, batch_size, seq_len, features)
        tensor = torch.stack([
            module(tensor[idx].contiguous(), verbose=verbose and idx == 0)
            for idx, module in enumerate(self.genomes.values())
        ])
        return tensor

    def evaluate(self, tensor: Tensor, keys: Union[int, list[int]] = None, verbose: int = None):
        logits = self.forward(tensor, keys=keys, verbose=verbose)
        return torch.argmax(logits, dim=-1)

    def guess(self, tensor, keys: Union[int, list[int]] = None):
        logits = self.forward(tensor, keys=keys)
        dist = torch.distributions.Categorical(F.softmax(logits, -1))
        return dist.sample()


if __name__ == "__main__":
    # -------------------- Globals -------------------- #
    DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    DIM_SIZE = 16
    LAYERS = 1
    BIAS = True
    NORM_GROUPS = 1
    EPSILON = 1e-12


    # -------------------- Dummy data -------------------- #
    TEST_CLASSES = 10
    INPUT_DIMS = 5
    OUTPUT_DIMS = TEST_CLASSES
    TEST_GENOMES = 3
    TEST_BATCHSIZE = 32
    TEST_SEQ_LEN = 8
    test_sequence = F.softmax(torch.normal(0, 1, (TEST_GENOMES, TEST_BATCHSIZE, INPUT_DIMS, TEST_SEQ_LEN), device=DEVICE, dtype=DTYPE), dim=-1)
    test_labels = test_sequence.argmax(dim=-1).to(torch.long)
    print(f"Test logits shape: {test_sequence.shape}, dtype: {test_sequence.dtype}, device: {test_sequence.device}")
    print(f"Test labels shape: {test_labels.shape}, dtype: {test_labels.dtype}, device: {test_labels.device}")


    # -------------------- Modules -------------------- #
    NEAT_MODULE = NeatNetwork(INPUT_DIMS, OUTPUT_DIMS, DIM_SIZE, 3, LAYERS, NORM_GROUPS, EPSILON, BIAS, DEVICE, DTYPE)
    CONFIG = meat.Config("convolution_test")
    CONFIG.genome.weight_init_mean = 0
    CONFIG.genome.weight_init_std = 0.1
    POPULATION = meat.Population(TEST_GENOMES, NEAT_MODULE, CONFIG, init_reporter=True)
    TORCH_MODULE = TorchNetwork(TEST_GENOMES, INPUT_DIMS, OUTPUT_DIMS, DIM_SIZE, 3, LAYERS, NORM_GROUPS, EPSILON, BIAS, DEVICE, DTYPE)
    TORCH_MODULE.load_weights(NEAT_MODULE)

    
    # -------------------- Comparing Outputs -------------------- #
    with torch.no_grad():
        neat_outputs = NEAT_MODULE(test_sequence)
        print(f"Neat outputs: \n{CM(neat_outputs[:2, :5, :3], Fore.LIGHTCYAN_EX)}, \n\tshape: {CM(neat_outputs.shape, Fore.LIGHTGREEN_EX)}")
        torch_outputs = TORCH_MODULE(test_sequence, verbose=False)
        print(f"Torch outputs: \n{CM(torch_outputs[:2, :5, :3], Fore.LIGHTCYAN_EX)}, \n\tshape: {CM(torch_outputs.shape, Fore.LIGHTGREEN_EX)}")