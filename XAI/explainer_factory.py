from XAI.reconstruction_explainer import ReconstructionExplainer
from XAI.mae_patch_explainer import MAEPatchExplainer
from XAI.gradcam_explainer import GradCAMExplainer
from XAI.integrated_gradients_explainer import IntegratedGradientsExplainer
from XAI.occlusion_explainer import OcclusionExplainer


class ExplainerFactory:
    """Factory for model-aware Terra-AId XAI explainers."""

    @staticmethod
    def create(model_type: str, method: str | None = None, **kwargs):
        model_type = str(model_type or "").lower()
        method = str(method or "auto").lower()

        if method in {"integrated_gradients", "ig"}:
            return IntegratedGradientsExplainer(**kwargs)
        
        if method == "occlusion":
            return OcclusionExplainer(**kwargs)
        
        if method in {"reconstruction", "reconstruction_error"}:
            return ReconstructionExplainer()
        
        if method in {"mae_patch", "patch", "mae"}:
            return MAEPatchExplainer(**kwargs)
        
        if method in {"gradcam", "grad_cam"}:
            return GradCAMExplainer()

        if model_type in {"mae", "mae_vit"}:
            return MAEPatchExplainer(**kwargs)
        
        if model_type in {"cnn_autoencoder", "classic_cnn", "classic_cnn_autoencoder"}:
            return ReconstructionExplainer()
        
        if model_type in {"resnet_autoencoder", "resnet50", "resnet50_autoencoder"}:
            return GradCAMExplainer()

        raise ValueError(f"No explainer available for model type: {model_type}")
