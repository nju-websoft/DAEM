import torch

def make_attention_mask(max_size, shade):
    attn_mask = torch.eye(max_size)
    for i in range(1, min(shade + 1, max_size - 1)):
        attn_mask[i:, :-i] += torch.eye(max_size - i)
        attn_mask[:-i, i:] += torch.eye(max_size - i)
    attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
    return attn_mask