import torch

from XAI.base_explainer import ExplainerBase


class ReconstructionExplainer(ExplainerBase):
    """Pixel/spatial reconstruction-error explainer for autoencoders."""

    def explain(self, model, x):
        model.eval()

        with torch.no_grad():
            output = model.predict(x)

        reconstruction = output["reconstruction"]
        error_map = torch.abs(x - reconstruction)
        spatial_error_map = error_map.mean(dim=1, keepdim=True)
        anomaly_score = spatial_error_map.mean(dim=[1, 2, 3])

        return {
                "method": "reconstruction_error",
                "reconstruction": reconstruction.detach(),
                "error_map": spatial_error_map.detach(),
                "channel_error_map": error_map.detach(),
                "anomaly_score": anomaly_score.detach(),
        }
