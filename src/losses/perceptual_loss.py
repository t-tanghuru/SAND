from typing import Dict, Tuple

import torch
from lpips import LPIPS


class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss based on LPIPS. Supports 2D and 2.5D (fake-3D) approaches.
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        is_fake_3d: bool = True,
        drop_ratio: float = 0.0,
        fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = True,
        spatial: bool = False,
    ):
        super(PerceptualLoss, self).__init__()

        if not (dimensions in [2, 3]):
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": spatial,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )
        self.fake_3D_views = (
            (
                []
                + ([((0, 2, 1, 3, 4), (1, 3, 4))] if 2 in fake_3d_axis else [])
                + ([((0, 3, 1, 2, 4), (1, 2, 4))] if 3 in fake_3d_axis else [])
                + ([((0, 4, 1, 2, 3), (1, 2, 3))] if 4 in fake_3d_axis else [])
            )
            if is_fake_3d
            else None
        )
        self.keep_ratio = 1 - drop_ratio
        self.lpips_normalize = lpips_normalize
        self.perceptual_function = (
            LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None
        )
        self.perceptual_factor = 1

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y = y.float()
        y_pred = y_pred.float()

        if self.dimensions == 3 and self.fake_3D_views:
            loss = torch.zeros(())
            for idx, fake_views in enumerate(self.fake_3D_views):
                loss = (
                    self._calculate_fake_3d_loss(
                        y=y,
                        y_pred=y_pred,
                        permute_dims=fake_views[0],
                        view_dims=fake_views[1],
                    )
                    * self.perceptual_factor
                )
        else:
            loss = (
                self.perceptual_function.forward(y, y_pred, normalize=self.lpips_normalize)
                * self.perceptual_factor
            )

        return loss

    def _calculate_fake_3d_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        permute_dims: Tuple[int, int, int, int, int],
        view_dims: Tuple[int, int, int],
    ):
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(-1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]])
        )
        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(-1, y_pred.shape[view_dims[0]], y_pred.shape[view_dims[1]], y_pred.shape[view_dims[2]])
        )
        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
            : int(y_pred_slices.shape[0] * self.keep_ratio)
        ]
        y_pred_slices = y_pred_slices.as_tensor()[indices]
        y_slices = y_slices.as_tensor()[indices]
        p_loss = torch.mean(
            self.perceptual_function.forward(y_slices, y_pred_slices, normalize=self.lpips_normalize)
        )
        return p_loss
