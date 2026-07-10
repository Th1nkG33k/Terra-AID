import yaml
import torch

from pathlib import Path
from Core.Managers.path_manager import PathManager
from Core.Managers.config_manager import ConfigManager, ModelConfig


# ============================================================
#    MODEL MANAGER
#
#    Model-facing service for the app.
#    ConfigManager is the single source of truth for loading YAML.
#    ModelManager consumes ModelConfig objects and exposes model
#    operations to the UI, trainers, and prediction code.
# ============================================================
class ModelManager:

    def __init__(self, configs_root: str | Path = None, config_manager: ConfigManager | None = None):

        self.config_manager = config_manager

        if self.config_manager is not None:
            self.pm = self.config_manager.pm
            self.configs_root = self.pm.MODEL_CONFIGS
            self.models = self.config_manager.models

        else:
            self.pm = PathManager()
            self.configs_root = Path(configs_root) if configs_root else self.pm.MODEL_CONFIGS
            self.config_manager = ConfigManager(self.pm)
            self.models = self.config_manager.models

        print(f"[ModelManager] Using model configs from: {self.configs_root}")


    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def list_models(self) -> list[str]:
        return list(self.models.keys())


    # ---------------------------------------------------------
    # Return UI-ready model summaries.
    # ---------------------------------------------------------
    def list_model_options(self) -> list[dict]:

        options = []

        for name, cfg in self.models.items():
            options.append({"name": cfg.model_name,
                            "key": name,
                            "stage": cfg.stage,
                            "architecture": getattr(cfg.architecture, "type", None),
                            "training_dataset": cfg.training_dataset,
                            "num_channels": self._runtime_channel_count(cfg),
                            "input_profile": self._runtime_profile_name(cfg),
            })

        return options


    def get(self, name: str) -> ModelConfig | None:
        return self.models.get(name)




    def _training_dataset_cfg(self, model_cfg):
        ds_name = getattr(model_cfg, "training_dataset", None)
        return self.config_manager.get_dataset(ds_name) if ds_name else None

    def _runtime_channel_count(self, model_cfg):
        ds = self._training_dataset_cfg(model_cfg)
        return getattr(ds, "num_input_channels", None) if ds is not None else None

    def _runtime_profile_name(self, model_cfg):
        count = self._runtime_channel_count(model_cfg)
        return f"derived_{count}ch" if count else None
    def reload(self):

        self.config_manager.reload()
        self.models = self.config_manager.models


    # ---------------------------------------------------------
    # Set the training dataset and copy its profile into the model.
    # ---------------------------------------------------------
    def set_training_dataset(self, model_name, dataset_name):

        cfg = self.get(model_name)

        if cfg is None:
            raise KeyError(f"Model '{model_name}' not found in loaded configs")

        cfg.set_training_dataset(dataset_name)

        dataset_cfg = self.config_manager.get_dataset(dataset_name)

        if dataset_cfg is not None:
            role = getattr(dataset_cfg, "role", "mixed")

            if role not in {"training", "mixed"}:
                raise ValueError(f"Dataset '{dataset_name}' has role '{role}' and cannot be used for training.")

            cfg.set_runtime_input_from_dataset(dataset_cfg)

        cfg.update_stage("training")
        self.reload()


    # ---------------------------------------------------------
    # Set the dataset used by prediction.
    # ---------------------------------------------------------
    def set_prediction_dataset(self, model_name, dataset_name):

        cfg = self.get(model_name)

        if cfg is None:
            raise KeyError(f"Model '{model_name}' not found in loaded configs")

        dataset_cfg = self.config_manager.get_dataset(dataset_name)

        if dataset_cfg is not None:
            role = getattr(dataset_cfg, "role", "mixed")

            role = {"prediction": "predictive", "validation": "predictive", "ground_truth": "predictive"}.get(role, role)

            if role not in {"predictive", "evaluation", "mixed"}:
                raise ValueError(f"Dataset '{dataset_name}' has role '{role}' and cannot be used for model prediction/evaluation.")

        cfg.set_prediction_dataset(dataset_name)
        self.reload()


    # ---------------------------------------------------------
    # Return a profile compatibility result.
    # ---------------------------------------------------------
    def check_dataset_compatibility(self, model_name: str, dataset_name: str) -> dict:

        model_cfg = self.get(model_name)
        dataset_cfg = self.config_manager.get_dataset(dataset_name)

        if model_cfg is None:
            return {"compatible": False, "reason": f"Model not found: {model_name}"}

        if dataset_cfg is None:
            return {"compatible": False, "reason": f"Dataset not found: {dataset_name}"}

        training_dataset_name = getattr(model_cfg, "training_dataset", None)
        training_dataset_cfg = self.config_manager.get_dataset(training_dataset_name) if training_dataset_name else None

        if training_dataset_cfg is None:
            return {"compatible": False,
                    "reason": "model has no training dataset selected",
                    "model_profile": None,
                    "dataset_profile": f"derived_{dataset_cfg.num_input_channels}ch",
                    "model_channels": None,
                    "dataset_channels": dataset_cfg.num_input_channels,
                    "model_channel_names": [],
                    "dataset_channel_names": dataset_cfg.input_channels,
            }

        model_channel_names = list(getattr(training_dataset_cfg, "input_channels", []) or [])
        dataset_channel_names = list(getattr(dataset_cfg, "input_channels", []) or [])
        model_channels = len(model_channel_names)
        dataset_channels = len(dataset_channel_names)

        channel_count_match = model_channels == dataset_channels
        channel_names_match = model_channel_names == dataset_channel_names
        compatible = bool(channel_count_match and channel_names_match)

        if compatible:
            reason = "derived input channels match"
        elif not channel_count_match:
            reason = "channel counts do not match"
        else:
            reason = "channel names/order do not match"

        return {"compatible": compatible,
                "model_profile": f"derived_{model_channels}ch",
                "dataset_profile": f"derived_{dataset_channels}ch",
                "model_channels": model_channels,
                "dataset_channels": dataset_channels,
                "model_channel_names": model_channel_names,
                "dataset_channel_names": dataset_channel_names,
                "reason": reason,
        }


    # ---------------------------------------------------------
    # Build a model config dictionary from create-model UI values.
    #
    #    Build a model configuration from the Create Model page.
    #
    #    Important: the UI keys use the CMC prefix. The previous version read
    #    CMD keys, so most user-entered values were missed and the hard-coded
    #    defaults were written instead.
    # ---------------------------------------------------------
    def _build_model_config_from_values(self, values: dict, model_name: str) -> dict:

        def _value(key, default=None):

            value = values.get(key, default)
            return default if value in (None, "") else value

        def _int(key, default=0):

            try:
                return int(_value(key, default))
            
            except (TypeError, ValueError):
                return default

        def _float(key, default=0.0):

            try:
                return float(_value(key, default))
            
            except (TypeError, ValueError):
                return default

        def _bool(key, default=False):

            value = _value(key, default)
            
            if isinstance(value, bool):
                return value
            
            return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}

        arc_type = str(_value("-CMC_ARC_TYPE-", "mae")).strip().lower()

        if arc_type in {"", "select model"}:
            raise ValueError("Please choose a model architecture before creating the model.")

        # Channel count/order are derived from the selected training dataset at runtime.
        architecture = {"type": arc_type}

        if arc_type == "mae":
            architecture.update({
                                "base_channels": _int("-CMC_ARC_BASE_CHANNELS-", 32),
                                "encoder_depth": _int("-CMC_ARC_ENCODER_DEPTH-", 5),
                                "decoder_depth": _int("-CMC_ARC_DECODER_DEPTH-", 3),
                                "embed_dim": _int("-CMC_ARC_EMBED_DIM-", 128),
                                "decoder_dim": _int("-CMC_ARC_DECODER_DIM-", 64),
                                "mask_ratio": _float("-CMC_ARC_MASK_RATIO-", 0.75),
            })

        elif arc_type == "cnn_autoencoder":
            architecture.update({
                                "base_channels": _int("-CMC_CNN_BASE_CHANNELS-", 32),
                                "encoder_depth": _int("-CMC_CNN_DEPTH-", 4),
                                "decoder_depth": _int("-CMC_CNN_DEPTH-", 4),
                                "latent_channels": _int("-CMC_CNN_LATENT_CHANNELS-", 256),
            })

        elif arc_type == "resnet_autoencoder":
            architecture.update({
                                "backbone": _value("-CMC_RESNET_BACKBONE-", "resnet50"),
                                "pretrained": _bool("-CMC_RESNET_PRETRAINED-", False),
                                "freeze_encoder_epochs": _int("-CMC_RESNET_FREEZE_EPOCHS-", 0),
            })

        else:
            raise ValueError(f"Unknown architecture type: {arc_type}")

        return {"model": {"name": model_name,
                          "stage": "created",
                          "device": "cuda" if values.get("-CMC_GPU-") else "cpu",
                },

                "architecture": architecture,

                # Shared training controls. These are intentionally architecture-
                # independent because your current trainers use the same training
                # loop, optimiser, scheduler and early stopping fields for all
                # reconstruction architectures.
                "optimizer": {"type": _value("-CMC_TRAIN_OPTIMIZER-", "AdamW"),
                              "lr": _float("-CMC_TRAIN_LR-", 0.0001),
                              "weight_decay": _float("-CMC_TRAIN_WD-", 0.00001),
                },

                "scheduler": {"type": _value("-CMC_ARCH_SCHEDULER-", "None"),
                              "warmup_epochs": _int("-CMC_ARCH_WARMUP-", 0),
                },

                "training": {"batch_size": _int("-CMC_TRAIN_BATCH-", 1),
                             "num_workers": _int("-CMC_TRAIN_WORKERS-", 0),
                             "epochs": _int("-CMC_TRAIN_EPOCHS-", 20),
                             "early_stopping_patience": _int("-CMC_TRAIN_PATIENCE-", 5),
                             "dataset": None,
                },

                "prediction": {"dataset": None,
                               "prediction_preset": "Balanced",
                               "threshold_metric": "fp_penalised_f1",
                               "false_positive_penalty": 0.2,
                               "max_false_positive_rate": 0.45,
                               "min_recall": 0.20,
                               "ignore_score_channels": ["SCL", "QC"],
                               "min_component_pixels": 0,
                },

                "paths": {"root": f"Data/Models/{model_name}",
                          "checkpoints": "Checkpoints",
                          "logs": "logs",
                          "outputs": "Visuals",
                },

                "provenance": {"created_by": "Terra-AId",
                               "version": "2.0",
                },
        }


    # ---------------------------------------------------------
    # Create the model folder structure and save the config twice.
    # ---------------------------------------------------------
    def create_model_from_values(self, values: dict) -> ModelConfig:

        model_name = (values.get("-CMC_MODEL_NAME-") or "").strip()

        if not model_name:
            raise ValueError("Model name is required.")

        cfg = self._build_model_config_from_values(values, model_name)

        model_root = self.pm.MODELS_ROOT / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        for folder in ["Checkpoints", "Visuals", "logs"]:
            (model_root / folder).mkdir(parents=True, exist_ok=True)

        project_config_path = model_root / f"{model_name}.yaml"
        central_config_path = self.pm.MODEL_CONFIGS / f"{model_name}.yaml"
        central_config_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_cfg = ModelConfig(model_name, cfg, central_config_path, self.pm)
        clean_yaml = tmp_cfg.to_yaml()
        project_config_path.write_text(clean_yaml, encoding="utf-8")
        central_config_path.write_text(clean_yaml, encoding="utf-8")

        self.reload()

        created_cfg = self.get(model_name)
        
        if created_cfg is None:
            created_cfg = ModelConfig(model_name, cfg, central_config_path, self.pm)
            self.models[model_name] = created_cfg

        print(f"[ModelManager] Model '{model_name}' created.")
        return created_cfg


    # ---------------------------------------------------------
    # Load a trained model from the model checkpoint folder.
    # ---------------------------------------------------------
    def load_model(self, model_or_name):

        cfg = self.get(model_or_name) if isinstance(model_or_name, str) else model_or_name

        if cfg is None:
            raise KeyError(f"Model '{model_or_name}' not found in loaded configs")

        ckpt_path = Path(cfg.paths.checkpoints) / f"{cfg.model_name}.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"[ModelManager] Checkpoint not found: {ckpt_path}")

        model = cfg.build_model()
        state = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(state)
        model.eval()

        return model
