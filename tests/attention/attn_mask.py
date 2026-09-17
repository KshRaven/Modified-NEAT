import torch
import math

if __name__ == "__main__":
    # shape(..., seq_len, dim_size)
    q_len, k_len = 4, 6
    query = torch.randn(q_len, 3)
    key = torch.randn(k_len, 3)
    # shape(..., qeury_len, key_len)
    tensor = query @ key.transpose(-2, -1)
    print(f"Tensor => \n{tensor} \n\tshape = {tensor.shape}")
    diff = k_len - q_len
    mask = torch.triu(torch.ones_like(tensor, dtype=torch.bool), diagonal=1+diff)
    print(f"Mask => \n{mask} \n\tshape = {mask.shape}")
    masked_tensor = tensor.masked_fill_(mask, float('-inf'))
    print(f"Masked Tensor => \n{masked_tensor} \n\tshape = {masked_tensor.shape}")
    soft_tensor = torch.round(torch.softmax(masked_tensor, dim=-1), decimals=2)
    print(f"Masked Tensor => \n{soft_tensor} \n\tshape = {soft_tensor.shape}")