
from build.nn.base import handle_input_mask, handle_output_mask, handle_input_dims, handle_output_dims
from build.models.base import Linear, LayerNorm
from build.models.sub import BufferEncoding, BufferEmbedding, MultiHeadSelfAttention
from build.util.qol import manage_params

from torch import Tensor

import torch.nn as nn
# import torch.nn.functional as F
import torch


class MiniFormer(nn.Module):
    def __init__(
            self, inputs: int, outputs: int, embed_size: int, max_seq_len: int, layers: int, heads: int,
                 kv_heads: int = None, dropout: float = 0.1, bias=False,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
        super(MiniFormer, self).__init__()
        self.distribution   = manage_params(options, 'distribution', 'discrete')
        self.epsilon        = manage_params(options, 'epsilon', 1e-10)
        self.constant       = manage_params(options, 'constant', 10000)
        self.primary_activation     = manage_params(options, 'pri_actv', nn.SiLU())
        self.secondary_activation   = manage_params(options, 'sec_actv', None)

        # BUILD
        self.embed      = BufferEmbedding(inputs, embed_size, bias, device, dtype, self.primary_activation)
        self.encode     = BufferEncoding(max_seq_len, embed_size, bias, device, dtype, None)
        self.att_norm   = nn.ModuleList([
            LayerNorm(embed_size, self.epsilon, True, bias, device, dtype)
        ])
        self.attention  = nn.ModuleList([
            MultiHeadSelfAttention(max_seq_len, embed_size, heads, kv_heads, self.constant, True,
                                   bias, device, dtype, self.primary_activation)
            for _ in range(layers)
        ])
        self.dec_norm   = LayerNorm(embed_size, self.epsilon, True, bias, device, dtype)
        output_dim      = outputs if self.distribution != 'discrete' else 2 ** outputs
        self.decode     = Linear(embed_size, output_dim, bias, device, dtype, self.secondary_activation, False)
        self.dropout    = nn.Dropout(dropout)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.eval()

        # ATTRIBUTES
        self.seq_len = max_seq_len

    @property
    def genomes_total(self):
        return self.decode.genomes_num

    def forward(self, inputs: Tensor, idx: int = None, genome_mask: Tensor = None, verbose: int = None):
        tensor, squeeze = handle_input_dims(inputs, self.genomes_total)
        tensor, genome_mask = handle_input_mask(tensor, genome_mask, -2)
        if idx is not None:
            tensor = tensor[:, :idx+1]
        if verbose:
            print(f"\nMiniFormer Input =>\n{tensor}\n\tdim = {tensor.shape}")
        tensor = self.dropout(self.encode(self.embed(tensor, mask=genome_mask, verbose=verbose), verbose=verbose))
        for x, (norm, block) in enumerate(zip(self.att_norm, self.attention)):
            query = tensor if idx is None else tensor[:, idx:idx+1]
            tensor = self.dropout(block(norm(tensor), context=query, mask=genome_mask, verbose=verbose if x == 0 else False) + query)
        tensor = self.decode(self.dec_norm(tensor), mask=genome_mask)
        if squeeze:
            tensor = tensor.squeeze(0)
        if verbose:
            print(f"\nMiniFormer Output =>\n{tensor}\n\tdim = {tensor.shape}")
        tensor = handle_output_mask(tensor, genome_mask, -2)
        outputs = handle_output_dims(tensor, squeeze, self.genomes_total)
        if self.distribution == 'discrete':
            outputs = torch.argmax(outputs, -1)
        return outputs

    def infer(self, tensor: Tensor, pos_idx: int = None, genome_mask: Tensor = None, verbose=False):
        idx = self.seq_len+pos_idx if pos_idx and pos_idx < 0 else pos_idx
        tensor = self.forward(tensor, idx, genome_mask, verbose)
        return tensor
