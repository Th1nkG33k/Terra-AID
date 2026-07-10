import torch

from XAI.base_explainer import ExplainerBase


class OcclusionExplainer(ExplainerBase):
    """Patch occlusion sensitivity for reconstruction error."""

    def __init__(self, patch_size=16, stride=None, fill_value=0.0):

        self.patch_size = int(patch_size)
        self.stride = int(stride or patch_size)
        self.fill_value = float(fill_value)

    def explain(self, model, x):

        model.eval()
        b, _c, h, w = x.shape
        scores = torch.zeros((b, 1, h, w), device=x.device)
        counts = torch.zeros_like(scores)

        with torch.no_grad():
            base_recon = model.predict(x)["reconstruction"]
            base_error = ((base_recon - x) ** 2).mean(dim=[1, 2, 3])

            for y in range(0, h, self.stride):

                for x0 in range(0, w, self.stride):

                    y1 = min(y + self.patch_size, h)
                    x1 = min(x0 + self.patch_size, w)
                    x_occ = x.clone()
                    x_occ[:, :, y:y1, x0:x1] = self.fill_value
                    recon = model.predict(x_occ)["reconstruction"]
                    err = ((recon - x_occ) ** 2).mean(dim=[1, 2, 3])
                    delta = (err - base_error).view(b, 1, 1, 1)
                    scores[:, :, y:y1, x0:x1] += delta
                    counts[:, :, y:y1, x0:x1] += 1

        return {
                "method": "occlusion_reconstruction_sensitivity",
                "sensitivity_map": (scores / counts.clamp_min(1)).detach(),
                "baseline_error": base_error.detach(),
        }
