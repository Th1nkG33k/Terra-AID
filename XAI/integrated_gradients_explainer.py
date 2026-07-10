import torch

from XAI.base_explainer import ExplainerBase


class IntegratedGradientsExplainer(ExplainerBase):
    """Input attribution against reconstruction error using integrated gradients."""

    def __init__(self, steps=24):
        self.steps = int(steps)

    def explain(self, model, x):
        
        model.eval()
        baseline = torch.zeros_like(x)
        total_grad = torch.zeros_like(x)

        for alpha in torch.linspace(0.0, 1.0, self.steps, device=x.device):
            x_step = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
            output = model(x_step)
            recon = output[0] if isinstance(output, tuple) else output
            loss = torch.mean((recon - x_step) ** 2)
            grad = torch.autograd.grad(loss, x_step, retain_graph=False)[0]
            total_grad += grad

        attribution = (x - baseline) * total_grad / max(self.steps, 1)
        return {
            "method": "integrated_gradients_reconstruction_error",
            "attribution": attribution.detach(),
            "attribution_map": attribution.abs().mean(dim=1, keepdim=True).detach(),
        }
