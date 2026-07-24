import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TimeSeriesToHeatmap(nn.Module):

    def __init__(self, clip_percentage: float = 0.005, target_size: tuple = None):
        super(TimeSeriesToHeatmap, self).__init__()
        self.clip_percentage = clip_percentage
        self.target_size = target_size  

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = x.flatten().astype(float)
        low_clip = np.quantile(y, self.clip_percentage)
        high_clip = np.quantile(y, 1 - self.clip_percentage)
        x_clipped = np.clip(x, low_clip, high_clip)

        x_min, x_max = x_clipped.min(), x_clipped.max()
        if x_max - x_min > 0:
            heatmap = (x_clipped - x_min) / (x_max - x_min)
        else:
            heatmap = x_clipped - x_min

        heatmap = heatmap.T

        return heatmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # handle batch / single sample uniformly
        single = False
        if x.dim() == 2:  # (L, M)
            x = x.unsqueeze(0)
            single = True

        x_np = x.detach().cpu().numpy()  # (B, L, M)
        out_list = []
        for sample in x_np:
            heat = self.transform(sample)  # (M, L)
            out_list.append(heat)
        out_np = np.stack(out_list, axis=0)  # (B, M, L)

        out_tensor = torch.from_numpy(out_np).to(x.device).float()
        out_tensor = out_tensor.unsqueeze(1)   #add channel dim -> (B,1,M,L)

        if single:
            out_tensor = out_tensor.squeeze(0)
        return out_tensor
