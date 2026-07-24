import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes=None,
        channels_per_scale=None,
        dropout: float = 0.0,
    ):
        super(MultiScaleCNN, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.kernel_sizes = kernel_sizes

        n_scales = len(kernel_sizes)
        if channels_per_scale is None:
            base = out_channels // n_scales
            channels_per_scale = [base] * n_scales
            channels_per_scale[-1] = out_channels - base * (n_scales - 1)
        elif isinstance(channels_per_scale, int):
            channels_per_scale = [channels_per_scale] * n_scales
        assert len(channels_per_scale) == n_scales, "channels_per_scale must be same as kernel_sizes "

        self.scales = nn.ModuleList()
        for k, c in zip(kernel_sizes, channels_per_scale):
            padding = (k - 1) // 2
            self.scales.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, c, kernel_size=k, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(c),
                )
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv2d(sum(channels_per_scale), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.scales:
            res.append(conv(x))
        y = torch.cat(res, dim=1)
        y = self.dropout(y)
        return self.proj(y)



