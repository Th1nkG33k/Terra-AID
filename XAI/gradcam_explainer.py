import torch
import torch.nn.functional as F

from XAI.base_explainer import ExplainerBase


class GradCAMExplainer(ExplainerBase):
    """
    Grad-CAM style explainer for reconstruction models with convolutional
    encoders, especially the ResNet autoencoder. The target is the model's
    reconstruction error rather than a classifier logit.
    """

    def explain(self, model, x, class_idx=None):

        model.eval()

        target_layer = model.get_xai_target_layer()
        if target_layer is None:
            raise ValueError("Grad-CAM requires model.get_xai_target_layer() to return a convolutional layer.")

        activations = []
        gradients = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        f_handle = target_layer.register_forward_hook(forward_hook)
        b_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            model.zero_grad(set_to_none=True)
            output = model(x)
            recon = output[0] if isinstance(output, tuple) else output
            target = torch.mean((recon - x) ** 2)
            target.backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            acts = activations[-1]
            grads = gradients[-1]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

            # Normalise each sample to [0, 1].
            flat = cam.flatten(1)
            cam_min = flat.min(dim=1).values.view(-1, 1, 1, 1)
            cam_max = flat.max(dim=1).values.view(-1, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return {
                    "method": "gradcam_reconstruction_error",
                    "heatmap": cam.detach(),
                    "reconstruction": recon.detach(),
                    "target_loss": target.detach(),
                    "class_idx": class_idx,
            }
        
        finally:
            f_handle.remove()
            b_handle.remove()
            model.zero_grad(set_to_none=True)
