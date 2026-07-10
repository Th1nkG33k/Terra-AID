# ---------------------------------------------------------------------
# Architecture registry for Terra-AId reconstruction models.

# Every model built here should behave as a reconstruction anomaly detector:
#     output = model(x)
# where output is either:
#     recon
#     (recon, metadata)
# The training and visualisation code normalises both forms.
# ---------------------------------------------------------------------

from Models.mae import build_mae
from Models.cnn_autoencoder import build_cnn_autoencoder
from Models.resnet_autoencoder import build_resnet_autoencoder


BUILDERS = {
            "mae": build_mae,
            "mae_vit": build_mae,
            "cnn_autoencoder": build_cnn_autoencoder,
            "classic_cnn": build_cnn_autoencoder,
            "classic_cnn_autoencoder": build_cnn_autoencoder,
            "resnet_autoencoder": build_resnet_autoencoder,
            "resnet50_autoencoder": build_resnet_autoencoder,
}


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model(cfg):
    # ---------------------------------------------------------------------
    # Build a Terra-AId model from either a ModelConfig instance or an
    # architecture dict/SimpleNamespace.
    # ---------------------------------------------------------------------
    architecture = getattr(cfg, "architecture", cfg)
    model_type = str(_get(architecture, "type", "mae")).lower()

    if model_type not in BUILDERS:
        known = ", ".join(sorted(BUILDERS.keys()))
        raise ValueError(f"Unknown model type: {model_type}. Known model types: {known}")

    return BUILDERS[model_type](architecture)
