import torch
import torch.nn.functional as F

from XAI.base_explainer import ExplainerBase


class MAEPatchExplainer(ExplainerBase):
    """MAE explainer using reconstruction error aggregated into patches."""

    def __init__(self, patch_size=16):
        self.patch_size = int(patch_size)

    def explain(self, model, x):

        model.eval()

        with torch.no_grad():
            output = model.predict(x)

        reconstruction = output["reconstruction"]
        error_map = torch.abs(x - reconstruction)
        pixel_error_map = error_map.mean(dim=1, keepdim=True)
        patch_error = self._patchify_error(pixel_error_map)
        mask = output.get("mask")

        result = {
                    "method": "mae_patch_reconstruction_error",
                    "reconstruction": reconstruction.detach(),
                    "pixel_error_map": pixel_error_map.detach(),
                    "patch_error_map": patch_error.detach(),
                    "anomaly_score": pixel_error_map.mean(dim=[1, 2, 3]).detach(),
        }

        if torch.is_tensor(mask):
            result["mask"] = mask.detach()
            
        return result

    def _patchify_error(self, pixel_error_map):
        """Convert [B, 1, H, W] pixel error into [B, 1, h, w] patch error."""
        h, w = pixel_error_map.shape[-2:]
        kernel = max(1, min(self.patch_size, h, w))
        return F.avg_pool2d(pixel_error_map, kernel_size=kernel, stride=kernel)
