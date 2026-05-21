"""Temporal satellite CNN — yield (kg/ha) + MC-dropout uncertainty.

Input contract (matches ``MaizeYieldDataset.__getitem__``):
    x : (B, T, H, W, 14)   torch.float32, channels-last

The forward pass permutes internally to ``(B, 14, T, H, W)`` for ``Conv3D``
(treating T as the depth axis), runs four 3D-conv blocks, global-average-
pools the resulting feature map, and projects to a single yield scalar via
a two-layer MLP. Output is ``(B,)`` so it matches the dataset's target
shape without broadcasting surprises.

Uncertainty is MC-dropout (Gal & Ghahramani 2016): keep dropout active at
inference, run N stochastic forward passes, report mean + std. The
``predict_with_uncertainty`` helper enables only the dropout layers — batch-
norm is held in eval mode so its running statistics aren't perturbed.

Sizing target: thesis-scale, ~700k parameters. With RALS 2018/19 giving
labels in the low hundreds, anything much bigger overfits before it
generalises.
"""
from __future__ import annotations

import torch
from torch import nn

import config


def _conv_block(
    in_ch: int,
    out_ch: int,
    dropout: float,
    pool: tuple[int, int, int] | None,
) -> nn.Sequential:
    """Conv3D(k=3) -> BN -> ReLU -> Dropout3d [-> MaxPool3d].

    ``Dropout3d`` zeros whole feature maps rather than individual voxels —
    this is the "spatial dropout" variant standard for MC-dropout in CNNs
    and is the noise source that drives the epistemic-uncertainty estimate.
    """
    layers: list[nn.Module] = [
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=dropout),
    ]
    if pool is not None:
        layers.append(nn.MaxPool3d(kernel_size=pool))
    return nn.Sequential(*layers)


class SatelliteCNN(nn.Module):
    """3D-Conv temporal CNN over Sentinel-2 stacks, MC-dropout uncertainty.

    Args:
        in_channels: input band count. Defaults to ``config.N_BANDS`` (14).
        base_channels: width of the first conv block. The subsequent blocks
            grow as base*2, base*4, base*4 — keeps the parameter count near
            ~700k at the default ``base_channels=32``.
        conv_dropout: Dropout3d rate inside conv blocks. Drives the MC
            variance — 0.15 is a thesis-grade default; raise it to widen
            uncertainty intervals at the cost of point accuracy.
        head_dropout: Dropout rate in the regression head's hidden layer.
    """

    def __init__(
        self,
        in_channels: int = config.N_BANDS,
        base_channels: int = 32,
        conv_dropout: float = 0.15,
        head_dropout: float = 0.30,
    ):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
        )

        # Pool schedule:
        #   block1: (1,2,2) — preserve T, halve spatial.  Phenology stays full-res
        #   block2: (2,2,2) — start compressing time once we have richer features
        #   block3: (2,2,2)
        #   block4: no pool — final receptive field, then global-pool.
        self.block1 = _conv_block(in_channels, c1, conv_dropout, pool=(1, 2, 2))
        self.block2 = _conv_block(c1, c2, conv_dropout, pool=(2, 2, 2))
        self.block3 = _conv_block(c2, c3, conv_dropout, pool=(2, 2, 2))
        self.block4 = _conv_block(c3, c4, conv_dropout, pool=None)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(c4, c4 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(c4 // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict yield (kg/ha).

        Args:
            x: (B, T, H, W, C) float tensor.
        Returns:
            (B,) float tensor — yield in kg/ha.
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input (B,T,H,W,C); got shape {tuple(x.shape)}"
            )
        # (B, T, H, W, C) -> (B, C, T, H, W) for Conv3d (channels-first, T=depth)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).flatten(1)        # (B, c4)
        x = self.head(x).squeeze(-1)              # (B,)
        return x

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-dropout: mean + std over ``n_samples`` stochastic forward passes.

        Holds batchnorm in ``eval`` (running stats) but re-enables dropout so
        each pass draws a different sub-network — the variance over passes
        is the epistemic-uncertainty estimate reported in the thesis.

        Args:
            x: (B, T, H, W, C) float tensor.
            n_samples: number of stochastic forward passes. 30 is the
                community default — pushes std error of the std estimate
                below ~15%.
        Returns:
            (mean, std), both shape (B,).
        """
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 to estimate std")
        prev_mode = self.training
        self.eval()
        self._enable_dropout_only()
        try:
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        finally:
            self.train(prev_mode)
        return preds.mean(dim=0), preds.std(dim=0)

    def _enable_dropout_only(self) -> None:
        """Re-enable dropout layers without flipping batchnorm back to train."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.train()


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = SatelliteCNN()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 25, 64, 64, 14)

    y = model(x)
    print(f"params: {n_params:,}")
    print(f"forward:   in={tuple(x.shape)}  ->  out={tuple(y.shape)}")
    print(f"  yields (untrained): {y.tolist()}")

    mean, std = model.predict_with_uncertainty(x, n_samples=30)
    print(f"MC-dropout (n=30):  mean={tuple(mean.shape)}, std={tuple(std.shape)}")
    for i in range(x.shape[0]):
        print(f"  sample {i}: mean={mean[i].item():+.3f}  std={std[i].item():.3f}")


if __name__ == "__main__":
    _smoke_test()
