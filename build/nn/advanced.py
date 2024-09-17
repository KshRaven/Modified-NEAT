
from build.nn.base import NeatModule
from build.nn.support import BufferEncoding, BufferEmbedding, ReFormEncoder
from build.util.qol import manage_params
from build.util.fancy_text import CM, Fore

from torch import Tensor

import torch
import torch.nn as nn


class Reformer(nn.Module):
    def __init__(self, genomes: int, inp_features: int, out_features: int, seq_len: int, embed_size: int, layers: int, 
                 heads: int, kv_heads: int = None, fwd_exp: int = None, dropout: float = None,
                 constant=10000.,  bias=False,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **ex):
        super(Reformer, self).__init__()
        epsilon = manage_params(ex, 'epsilon', 1e-12)
        actv = manage_params(ex, 'activation', None)

        # BUILD
        self.embed      = BufferEmbedding(genomes, inp_features, embed_size, bias, device, dtype)
        self.encode     = BufferEncoding(seq_len, embed_size, device, dtype)
        self.transform  = ReFormEncoder(
            genomes, seq_len, embed_size, heads, layers, fwd_exp, kv_heads, dropout, epsilon, constant, 
            bias, device, dtype
        )
        self.norm       = nn.LayerNorm(embed_size, epsilon, bias=bias, device=device, dtype=dtype)
        # self.post_norm  = RMSNormalization(embed_size, epsilon, device, dtype)
        self.flatten    = nn.Flatten(-2, -1)
        self.decode     = NeatModule(genomes, embed_size * seq_len, out_features, bias, 0, device, dtype)
        self.activate   = actv

        # ATTRIBUTES
        self.genome_num = genomes
        self.embed_size  = embed_size
        self.max_seq_len = seq_len
        self.heads       = heads
        self.kv_heads    = kv_heads
        self.layers      = layers
        self.inp_num     = inp_features
        self.out_num     = out_features

        # STATE
        self.device     = device
        self.dtype      = dtype

    def pad_sequence(self, tensor: Tensor):
        genomes, batch_size, seq_len, features = tensor.shape
        padding = torch.zeros(genomes, batch_size, self.max_seq_len - seq_len, features).to(self.device, self. dtype)
        return torch.cat([padding, tensor], dim=2)

    def forward(self, tensor: Tensor, debug: bool = False) -> Tensor:
        if debug:
            print(f"\n{CM('Reformer Input =>', Fore.LIGHTMAGENTA_EX)}\n{tensor}\n\tdim = {tensor.shape}")
        if tensor.shape[2] != self.max_seq_len:
            tensor = self.pad_sequence(tensor)
        # squeeze = False
        # if tensor.ndim == 2:
        #     tensor = tensor.unsqueeze(0)
        #     squeeze = True
        tensor = self.embed(tensor)
        if debug:
            print(f"\n{CM('Input Embedding =>', Fore.LIGHTCYAN_EX)}\n{tensor}\n\tdim = {tensor.shape}")
        tensor = self.encode(tensor)
        if debug:
            print(f"\n{CM('Input Encoding =>', Fore.LIGHTRED_EX)}\n{tensor}\n\tdim = {tensor.shape}")
        tensor = self.transform(tensor, debug)
        if debug:
            print(f"\n{CM('Encoder Output =>', Fore.LIGHTYELLOW_EX)}\n{tensor}\n\tdim = {tensor.shape}")
        tensor = self.decode(self.flatten(self.norm(tensor)))
        if self.activate is not None:
            tensor = self.activate(tensor)
        # if squeeze:
        #     tensor = tensor.squeeze(0)
        if debug:
            print(f"\n{CM('Reformer Output =>', Fore.LIGHTGREEN_EX)}\n{tensor}\n\tdim = {tensor.shape}")
        return tensor

    def learn(self, inputs: Tensor, debug=False):
        return self.forward(inputs, debug)

    def infer(self, inputs: Tensor, **ex):
        debug = manage_params(ex, 'debug', False)
        threshold = manage_params(ex, 'threshold', None)
        with torch.no_grad():
            torch.cuda.empty_cache()
            outputs = self.forward(inputs, debug)
            if threshold is not None:
                outputs = (outputs >= threshold).to(self.device, self.dtype)
            return outputs
