
import json
from pathlib import Path

from Core.Managers.training_manager import TrainingManager
from Core.Managers.prediction_manager import PredictionManager
from Core.Pytorch.pytorch_dataset_factory import PyTorchDatasetFactory
from Core.Utils.image_utility import ImageUtility


# ============================================================
#    MODEL TASKS
#
#    Background worker functions for model workflows.
#    These functions consume ModelManager and DatasetManager.
# ============================================================




# ---------------------------------------------------------
# Send a staged progress message if a worker is available.
# ---------------------------------------------------------
def _progress(worker, pct, message):

    if worker:
        worker.progress(f"{pct}% - {message}")


# ---------------------------------------------------------
# Train a reconstruction model on its selected dataset.
# ---------------------------------------------------------
def train_model_task(model_name, dataset_name, model_manager, dataset_manager, worker=None, **kwargs):

    cfg = model_manager.get(model_name)
    ds_cfg = dataset_manager.get(dataset_name)

    if cfg is None:
        raise RuntimeError(f"Model not found: {model_name}")

    if ds_cfg is None:
        raise RuntimeError(f"Dataset not found: {dataset_name}")

    _progress(worker, 5, "Loading training dataset")

    factory = PyTorchDatasetFactory()
    train_ds = factory.build(ds_cfg)
    val_ds = factory.build(ds_cfg)

    if len(train_ds) == 0:
        raise RuntimeError(f"Dataset '{dataset_name}' has no processed tile folders.")

    x0, meta0 = train_ds[0]
    real_depth = int(x0.shape[0])

    channel_names = []
    if isinstance(meta0, dict):
        channel_names = list(meta0.get("channel_names", []) or [])

    cfg.architecture.num_channels = real_depth
    cfg.architecture.channel_names = channel_names
    cfg.runtime_input_channels = channel_names

    if worker:
        if channel_names:
            worker.status(f"Detected {real_depth} input channels: {', '.join(channel_names)}")
        else:
            worker.status(f"Detected {real_depth} input channels from processed tile files")

    _progress(worker, 15, "Building model")
    model = cfg.build_model()

    _progress(worker, 20, "Starting training")
    trainer = TrainingManager()
    best_loss, log = trainer.train_one_MAE_config(model=model,
                                                  train_dataset=train_ds,
                                                  val_dataset=val_ds,
                                                  config=cfg.cfg,
                                                  save_dir=cfg.paths.checkpoints,
                                                  device=cfg.device,
                                                  worker=worker,
    )

    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    (cfg.paths.logs / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    _progress(worker, 95, "Saving trained model config")
    cfg.update_stage("trained")
    model_manager.reload()
    _progress(worker, 100, "Model training complete")

    return {"message": "Model trained successfully.",
            "best_loss": best_loss,
            "model": model_name,
            "show_view": "model",
    }


# ---------------------------------------------------------
# Run prediction if the model and dataset profiles match.
# ---------------------------------------------------------
def run_prediction_task(model_name, dataset_name, model_manager, dataset_manager, worker=None, **kwargs):

    _progress(worker, 5, "Checking prediction compatibility")
    compatibility = model_manager.check_dataset_compatibility(model_name, dataset_name)

    if not compatibility.get("compatible"):
        raise RuntimeError(
            "Prediction blocked because the model and dataset profiles do not match.\n"
            f"Model profile: {compatibility.get('model_profile')} "
            f"({compatibility.get('model_channels')} channels)\n"
            f"Dataset profile: {compatibility.get('dataset_profile')} "
            f"({compatibility.get('dataset_channels')} channels)"
        )

    cfg = model_manager.get(model_name)
    ds_cfg = dataset_manager.get(dataset_name)

    if cfg is None:
        raise RuntimeError(f"Model not found: {model_name}")

    if ds_cfg is None:
        raise RuntimeError(f"Dataset not found: {dataset_name}")

    # Runtime channel names come from the model's training dataset. They are not
    # duplicated in the model YAML.
    train_ds_name = getattr(cfg, "training_dataset", None)
    train_ds_cfg = dataset_manager.get(train_ds_name) if train_ds_name else None
    if train_ds_cfg is not None:
        cfg.architecture.num_channels = int(getattr(train_ds_cfg, "num_input_channels", 0) or 0)
        cfg.architecture.channel_names = list(getattr(train_ds_cfg, "input_channels", []) or [])

    _progress(worker, 15, "Preparing prediction outputs")
    predictor = PredictionManager(threshold_percentile=95.0)
    save_dir = cfg.paths.outputs / "predictions" / dataset_name

    summary = predictor.predict_dataset(model_cfg=cfg,
                                        dataset_cfg=ds_cfg,
                                        save_dir=save_dir,
                                        device=cfg.device,
                                        worker=worker,
    )

    _progress(worker, 100, "Prediction complete")

    return {"message": "Prediction completed successfully.",
            "output_dir": str(Path(save_dir)),
            "summary_path": str(Path(save_dir) / "prediction_summary.json"),
            "tile_count": summary["tile_count"],
            "dataset": dataset_name,
            "model": model_name,
            "show_view": "model",
    }


# ---------------------------------------------------------
# Generate trained-model visualisations.
# ---------------------------------------------------------
def model_visualisation_task(model_name, visual_type, model_manager, dataset_manager, worker=None, **kwargs):

    _progress(worker, 5, "Loading model config")
    cfg = model_manager.get(model_name)

    if cfg is None:
        raise RuntimeError(f"Model not found: {model_name}")

    dataset_name = cfg.training_dataset
    if not dataset_name:
        raise RuntimeError(f"Model '{model_name}' has no training dataset set.")

    ds_cfg = dataset_manager.get(dataset_name)
    if ds_cfg is None:
        raise RuntimeError(f"Dataset not found: {dataset_name}")

    image_util = ImageUtility()
    output_dir = cfg.paths.outputs / "visualisations"
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress(worker, 20, "Starting model visualisation")

    if visual_type == "anomaly_map":
        if worker:
            worker.status("Generating anomaly map...")

        out_path = image_util.generate_model_anomaly_map(model_cfg=cfg,
                                                         dataset_cfg=ds_cfg,
                                                         save_dir=output_dir,
                                                         worker=worker,
        )
        title = "Anomaly Map"

    elif visual_type == "clustering":
        if worker:
            worker.status("Generating clustering map...")

        out_path = image_util.generate_model_clustering_map(model_cfg=cfg,
                                                            dataset_cfg=ds_cfg,
                                                            save_dir=output_dir,
                                                            n_clusters=6,
                                                            worker=worker,
        )
        title = "Model Clustering"

    else:
        raise RuntimeError(f"Unsupported visualisation type: {visual_type}")

    _progress(worker, 100, f"{title} complete")

    return {"message": f"{title} generated successfully.",
            "output_path": str(out_path),
            "title": title,
            "model": model_name,
            "show_view": "model",
    }
