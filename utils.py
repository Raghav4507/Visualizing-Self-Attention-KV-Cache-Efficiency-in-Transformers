import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()

def plot_attention_grid(attn: torch.Tensor, figsize_per=(1.5, 1.2)):
    num_layers, num_heads, seq, _ = attn.shape
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * figsize_per[0], num_layers * figsize_per[1]))
    for l in range(num_layers):
        for h in range(num_heads):
            ax = axes[l, h] if num_layers > 1 else axes[h]
            mat = attn[l, h, :, :]
            im = ax.imshow(mat, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{l} H{h}", fontsize=8)
    plt.tight_layout()
    return fig
