"""Decision-level fusion of the satellite + ground branches.

Wraps a ``SatelliteCNN`` and a ``GroundCNN`` and produces a single fused
maize-yield estimate per field::

    y_fused = yield_sat + alpha * adjustment

where ``yield_sat`` comes from the satellite branch (kg/ha), ``adjustment``
is a yield delta (kg/ha) predicted by a small fusion MLP fed the ground
branch's ``embedding`` concatenated with its ``health_score``, and
``alpha`` is a single learnable scalar (``nn.Parameter``) initialised to
0.5 so the model starts close to satellite-only and grows / shrinks the
fusion contribution during training.

The fusion MLP itself is deterministic — all epistemic uncertainty is
inherited from MC-dropout inside the two branches. The
``predict_with_uncertainty`` method enables only Dropout layers on
**both** branches simultaneously and runs ``n_samples`` coherent forward
passes, so each MC draw is one self-consistent stochastic realisation of
the whole fusion network rather than a mix-and-match of branch posteriors.

``model.alpha`` is exposed as a property so ``train.py`` can log its
per-epoch trajectory — that plot is one of the methodology figures
called for in CLAUDE.md ("how much did the fusion learn to trust the
ground branch?").
"""
from __future__ import annotations

import torch
from torch import nn

from farmers_guide.models.ground_cnn import GroundCNN
from farmers_guide.models.satellite_cnn import SatelliteCNN


class YieldFusion(nn.Module):
    """Satellite + ground fusion module — outputs a single yield (kg/ha).

    Args:
        satellite_cnn: a constructed ``SatelliteCNN`` instance. Owned by
            the fusion module (registered as a submodule), so its weights
            train end-to-end with the fusion MLP and ``alpha``.
        ground_cnn: a constructed ``GroundCNN`` instance. Same ownership.
        hidden_dim: width of the fusion MLP's hidden layer.
        alpha_init: initial value of the learnable fusion weight. 0.5 keeps
            the model close to satellite-only at start; gradient descent
            chooses where to take it.
    """

    def __init__(
        self,
        satellite_cnn: SatelliteCNN,
        ground_cnn: GroundCNN,
        hidden_dim: int = 64,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        self.satellite = satellite_cnn
        self.ground = ground_cnn

        # Read the actual embedding width off the instance, not from a
        # constant — keeps fusion correct if the ground branch is later
        # constructed with a non-default ``embedding_dim``.
        embedding_dim = ground_cnn.embedding_head.out_features
        fusion_in = embedding_dim + 1   # +1 for the scalar health_score

        # No dropout in the fusion MLP — uncertainty comes from the branches.
        # Two layers, ReLU between them, scalar output (yield delta in kg/ha).
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self._alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    # -- public accessor ----------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """The learnable fusion weight (scalar tensor).

        Returned as the underlying ``nn.Parameter`` so gradients still flow
        through it; call ``model.alpha.item()`` to log the float value per
        epoch.
        """
        return self._alpha

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        sat_input: torch.Tensor,
        ground_input: torch.Tensor,
    ) -> torch.Tensor:
        """Single fused-yield forward pass.

        Args:
            sat_input: (B, T, H, W, 14) satellite cube patch (matches
                ``MaizeYieldDataset``'s ``patch`` output).
            ground_input: (B, 3, 224, 224) ImageNet-normalised RGB photo.
        Returns:
            (B,) float tensor — fused yield in kg/ha.
        """
        yield_sat = self.satellite(sat_input)                  # (B,)
        gout = self.ground(ground_input)
        embedding = gout["embedding"]                          # (B, D)
        health = gout["health_score"]                          # (B,)
        features = torch.cat([embedding, health.unsqueeze(-1)], dim=1)  # (B, D+1)
        adjustment = self.fusion_mlp(features).squeeze(-1)     # (B,)
        return yield_sat + self._alpha * adjustment

    # -- MC-dropout inference ----------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        sat_input: torch.Tensor,
        ground_input: torch.Tensor,
        n_samples: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paired MC-dropout: ``n_samples`` coherent stochastic fused yields.

        Each iteration runs **one** full forward pass through the wrapped
        fusion network with both branches' Dropout layers active. That
        single pass is one paired draw of ``(yield_sat, embedding,
        health)`` — the satellite and ground stochastic realisations are
        consumed by the same fusion step, so cross-branch dependencies
        are preserved in the resulting variance.

        Calling ``_enable_dropout_only`` separately on each branch and
        then composing their independent variances later would be
        incorrect: it would treat the branches as independent posteriors
        when in fact they are entered into the same downstream MLP.

        Args:
            sat_input: (B, T, H, W, 14) satellite cube patch.
            ground_input: (B, 3, 224, 224) phone photo.
            n_samples: number of stochastic forward passes (>=2).
        Returns:
            ``(mean, std)`` — both shape ``(B,)``, in kg/ha.
        """
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 to estimate std")

        prev_mode = self.training
        # eval() puts the whole fusion module — including BN layers in both
        # branches — into eval mode. Then re-enable dropout on both branches
        # in a single pass so the masks fire together inside each forward.
        self.eval()
        self.satellite._enable_dropout_only()
        self.ground._enable_dropout_only()
        try:
            preds = torch.stack(
                [self.forward(sat_input, ground_input) for _ in range(n_samples)],
                dim=0,
            )                                                  # (n_samples, B)
        finally:
            self.train(prev_mode)
        return preds.mean(dim=0), preds.std(dim=0)


def _smoke_test() -> None:
    torch.manual_seed(0)

    satellite = SatelliteCNN()
    ground = GroundCNN(pretrained=False)
    model = YieldFusion(satellite, ground)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"params: {n_params:,}  trainable: {n_trainable:,}  "
        f"alpha (init): {model.alpha.item():.4f}"
    )

    sat_in = torch.randn(2, 25, 64, 64, 14)
    ground_in = torch.randn(2, 3, 224, 224)

    y = model(sat_in, ground_in)
    print(
        f"forward:   sat={tuple(sat_in.shape)}  ground={tuple(ground_in.shape)}  "
        f"->  y_fused={tuple(y.shape)}"
    )
    print(f"  yields (untrained): {[f'{v:+.3f}' for v in y.tolist()]}")

    mean, std = model.predict_with_uncertainty(sat_in, ground_in, n_samples=30)
    print(
        f"MC-dropout (n=30):  mean={tuple(mean.shape)}  std={tuple(std.shape)}"
    )
    for i in range(sat_in.shape[0]):
        print(
            f"  sample {i}: mean={mean[i].item():+.3f}  std={std[i].item():.4f}"
        )
    print(f"alpha (still init, untrained): {model.alpha.item():.4f}")


if __name__ == "__main__":
    _smoke_test()
