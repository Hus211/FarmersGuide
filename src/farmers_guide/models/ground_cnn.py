"""MobileNetV2 transfer model on field photos -> embedding + health score.

Input contract (standard ImageNet shape):
    x : (B, 3, 224, 224)   torch.float32, channels-first, ImageNet-normalised

Forward returns a dict::

    {
        "embedding":    (B, 32),   # canopy-state embedding (no activation)
        "health_score": (B,),      # sigmoid-bounded, in [0, 1]
    }

The embedding feeds the decision-level ``fusion`` module; the health score is
the human-interpretable scalar surfaced in figures / per-field summaries.

Transfer setup: torchvision's ImageNet-pretrained MobileNetV2 backbone with
the first ~75% of feature blocks frozen. Smallholder field photos are
out-of-domain for ImageNet (close-range crop canopy vs. object-centric web
images), so the late blocks are fine-tuned while the low-level edge / texture
filters are kept fixed. The freeze fraction is exposed as a constructor
argument so it can be ablated for the methodology chapter.

Uncertainty mirrors ``satellite_cnn.SatelliteCNN``: ``nn.Dropout`` before the
heads, ``predict_with_uncertainty`` re-enables only Dropout layers (BN running
stats stay frozen). MC variance is reported separately for the embedding and
for the health score, since they may be propagated downstream differently.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


# torchvision MobileNetV2's `features` is a Sequential of 19 sub-modules
# (initial conv + 17 InvertedResidual + final conv). 1280 is the channel
# count of the final feature map.
_MOBILENET_V2_FEATURE_BLOCKS = 19
_MOBILENET_V2_OUT_CHANNELS = 1280


class GroundCNN(nn.Module):
    """MobileNetV2 transfer model -> 32-d embedding + scalar health score.

    Args:
        embedding_dim: width of the canopy-state embedding. 32 is the
            fusion-module contract; raise if downstream fusion needs more
            capacity.
        dropout: Dropout rate on the pooled feature vector before the heads.
            Drives MC-dropout variance.
        freeze_fraction: fraction of ``features`` blocks to freeze (count
            from the input). 0.75 freezes the first 14 of 19 blocks.
        pretrained: load ImageNet weights. Pass ``False`` for smoke tests or
            offline environments — random init is fine for shape checks.
    """

    EMBEDDING_DIM_DEFAULT = 32

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM_DEFAULT,
        dropout: float = 0.30,
        freeze_fraction: float = 0.75,
        pretrained: bool = True,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= freeze_fraction <= 1.0:
            raise ValueError("freeze_fraction must be in [0, 1]")

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features            # (B, 1280, 7, 7)

        # Freeze the first ``freeze_fraction`` of feature blocks. Late blocks
        # (block index >= n_freeze) keep ``requires_grad=True`` so the
        # high-level filters adapt to the smallholder-canopy domain.
        n_blocks = len(self.features)
        if n_blocks != _MOBILENET_V2_FEATURE_BLOCKS:
            # Defensive: torchvision could change the block count in a
            # future release and we'd silently freeze the wrong thing.
            raise RuntimeError(
                f"Unexpected MobileNetV2 layout: {n_blocks} feature blocks "
                f"(expected {_MOBILENET_V2_FEATURE_BLOCKS}). Re-check the "
                f"freeze logic before training."
            )
        n_freeze = int(round(n_blocks * freeze_fraction))
        for i, block in enumerate(self.features):
            if i < n_freeze:
                for p in block.parameters():
                    p.requires_grad = False
        self._n_frozen_blocks = n_freeze

        self.pool = nn.AdaptiveAvgPool2d(1)
        # MC-dropout point. Plain Dropout, not Dropout2d — at this stage the
        # input is the pooled (B, 1280) feature vector, not a feature map.
        self.dropout = nn.Dropout(p=dropout)

        # Two heads off the same pooled, dropout-perturbed feature. Sharing
        # the trunk this way means MC variance is consistent between
        # embedding and health-score readings — one stochastic forward pass
        # gives you one paired sample of both.
        self.embedding_head = nn.Linear(_MOBILENET_V2_OUT_CHANNELS, embedding_dim)
        self.health_head = nn.Linear(_MOBILENET_V2_OUT_CHANNELS, 1)

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a single (deterministic-when-eval) forward pass.

        Args:
            x: (B, 3, 224, 224) float tensor, ImageNet-normalised.
        Returns:
            ``{"embedding": (B, embedding_dim), "health_score": (B,)}``.
            ``health_score`` is sigmoid-bounded to [0, 1].
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Expected (B, 3, H, W) input; got shape {tuple(x.shape)}"
            )
        f = self.features(x)                  # (B, 1280, 7, 7)
        f = self.pool(f).flatten(1)           # (B, 1280)
        f = self.dropout(f)
        embedding = self.embedding_head(f)    # (B, embedding_dim)
        health = torch.sigmoid(
            self.health_head(f).squeeze(-1)
        )                                     # (B,)
        return {"embedding": embedding, "health_score": health}

    # -- MC-dropout inference ----------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> dict[str, torch.Tensor]:
        """MC-dropout: mean + std for both the embedding and health score.

        Args:
            x: (B, 3, 224, 224) float tensor.
            n_samples: number of stochastic forward passes.
        Returns:
            ``{"embedding_mean": (B, D), "embedding_std": (B, D),
               "health_mean":    (B,),   "health_std":    (B,)}``
        """
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 to estimate std")
        prev_mode = self.training
        self.eval()
        self._enable_dropout_only()
        try:
            embeddings: list[torch.Tensor] = []
            healths: list[torch.Tensor] = []
            for _ in range(n_samples):
                out = self.forward(x)
                embeddings.append(out["embedding"])
                healths.append(out["health_score"])
        finally:
            self.train(prev_mode)
        e = torch.stack(embeddings, dim=0)    # (n_samples, B, D)
        h = torch.stack(healths, dim=0)       # (n_samples, B)
        return {
            "embedding_mean": e.mean(dim=0),
            "embedding_std":  e.std(dim=0),
            "health_mean":    h.mean(dim=0),
            "health_std":     h.std(dim=0),
        }

    def _enable_dropout_only(self) -> None:
        """Re-enable Dropout layers without flipping BatchNorm to train.

        BatchNorm in MobileNetV2 is sensitive — letting it switch to
        batch-statistics mode at inference time changes outputs depending
        on batch composition and breaks single-image inference. Same
        pattern as ``satellite_cnn.SatelliteCNN``.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.train()


def _smoke_test() -> None:
    torch.manual_seed(0)
    # Skip the ImageNet weights download for the smoke test — shape checks
    # don't care about init quality.
    model = GroundCNN(pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"params: {n_params:,}  trainable: {n_trainable:,} "
        f"({n_trainable / n_params:.1%})  "
        f"frozen feature blocks: {model._n_frozen_blocks}/"
        f"{_MOBILENET_V2_FEATURE_BLOCKS}"
    )

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(
        f"forward:   in={tuple(x.shape)}  ->  "
        f"embedding={tuple(out['embedding'].shape)}, "
        f"health_score={tuple(out['health_score'].shape)}"
    )
    print(
        f"  health range: [{out['health_score'].min().item():.3f}, "
        f"{out['health_score'].max().item():.3f}]  (must be in [0, 1])"
    )

    mc = model.predict_with_uncertainty(x, n_samples=30)
    print(
        f"MC-dropout (n=30):  "
        f"emb_mean={tuple(mc['embedding_mean'].shape)} "
        f"emb_std={tuple(mc['embedding_std'].shape)} "
        f"health_mean={tuple(mc['health_mean'].shape)} "
        f"health_std={tuple(mc['health_std'].shape)}"
    )
    for i in range(x.shape[0]):
        print(
            f"  sample {i}: health_mean={mc['health_mean'][i].item():.3f} "
            f"health_std={mc['health_std'][i].item():.4f} "
            f"emb_std_mean={mc['embedding_std'][i].mean().item():.4f}"
        )


if __name__ == "__main__":
    _smoke_test()
