
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

def _dataset_visual_root(model_cfg, dataset_name):
    """Return the single per-dataset Visuals folder for a model run.

    All outputs for a dataset/model combination live under:
        Data/Models/<model>/Visuals/<dataset_name>/
    so the whole run can be exported by copying that folder.
    """

    safe_name = str(dataset_name or "unknown_dataset").strip() or "unknown_dataset"
    return model_cfg.paths.outputs / safe_name


def _apply_latest_calibration_to_model_cfg(model_cfg, dataset_name, worker=None):
    """Ensure prediction uses the latest calibration output for this model/dataset.

    Calibration writes a report file and should also persist the selected threshold
    into the model YAML.  In practice, prediction should not rely only on an
    already-loaded ModelConfig object because the UI/worker may still hold stale
    threshold values.  This helper reads the latest calibration_summary.json and
    applies the selected threshold to the runtime config immediately before
    prediction.
    """

    calibration_path = _dataset_visual_root(model_cfg, dataset_name) / "calibration" / "calibration_summary.json"

    if not calibration_path.exists():
        return False

    try:
        summary = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if worker:
            worker.status(f"Calibration summary could not be read; using model YAML threshold settings: {exc}")
        return False

    selected_value = summary.get("selected_threshold_value")
    selected_percentile = summary.get("selected_threshold_percentile")

    if selected_value in (None, "") or selected_percentile in (None, ""):
        return False

    selection = summary.get("selection", {}) or {}
    prediction_cfg = model_cfg.cfg.setdefault("prediction", {})

    previous_percentile = prediction_cfg.get("threshold_percentile")
    previous_value = prediction_cfg.get("threshold_value")

    prediction_cfg["prediction_preset"] = summary.get("prediction_preset", prediction_cfg.get("prediction_preset", "Balanced"))
    prediction_cfg["threshold_mode"] = "calibrated_global"
    prediction_cfg["threshold_value"] = float(selected_value)
    prediction_cfg["threshold_percentile"] = float(selected_percentile)
    prediction_cfg["threshold_metric"] = summary.get("metric_target") or selection.get("metric") or prediction_cfg.get("threshold_metric", "fp_penalised_f1")

    if summary.get("false_positive_penalty") is not None:
        prediction_cfg["false_positive_penalty"] = float(summary.get("false_positive_penalty"))
    elif selection.get("false_positive_penalty") is not None:
        prediction_cfg["false_positive_penalty"] = float(selection.get("false_positive_penalty"))

    if summary.get("max_false_positive_rate") is not None:
        prediction_cfg["max_false_positive_rate"] = float(summary.get("max_false_positive_rate"))
    elif selection.get("max_false_positive_rate") is not None:
        prediction_cfg["max_false_positive_rate"] = float(selection.get("max_false_positive_rate"))

    if summary.get("min_recall") is not None:
        prediction_cfg["min_recall"] = float(summary.get("min_recall"))
    elif selection.get("min_recall") is not None:
        prediction_cfg["min_recall"] = float(selection.get("min_recall"))

    if summary.get("min_component_pixels") is not None:
        prediction_cfg["min_component_pixels"] = int(summary.get("min_component_pixels") or 0)
    elif selection.get("min_component_pixels") is not None:
        prediction_cfg["min_component_pixels"] = int(selection.get("min_component_pixels") or 0)

    prediction_cfg["threshold_calibration_dataset"] = dataset_name
    prediction_cfg["threshold_calibration_summary"] = str(calibration_path)
    prediction_cfg["threshold_calibration_fallback_used"] = bool(summary.get("selection_fallback_used", False))
    prediction_cfg["threshold_calibration_eligible"] = bool(summary.get("selected_eligible_for_selection", True))

    try:
        model_cfg.save()
    except Exception as exc:
        if worker:
            worker.status(f"Calibration threshold applied for this run but could not be saved to YAML: {exc}")

    if worker:
        worker.status(
            f"Using calibrated threshold from latest calibration: p{float(selected_percentile):g} = {float(selected_value):.6f} "
            f"(previous YAML p{previous_percentile}, value {previous_value})"
        )

    return True



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
def run_prediction_task(model_name, dataset_name, model_manager, dataset_manager, worker=None, workflow="predictive", **kwargs):

    # Always reload before prediction so the task cannot use stale threshold
    # values from before the most recent calibration run.
    model_manager.reload()
    dataset_manager.reload()

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

    _apply_latest_calibration_to_model_cfg(cfg, dataset_name, worker=worker)

    workflow = str(workflow or "predictive").lower()
    if workflow not in {"predictive", "evaluation"}:
        workflow = "predictive"

    role = str(getattr(ds_cfg, "role", "mixed") or "mixed").lower()
    role = {"prediction": "predictive", "validation": "predictive", "ground_truth": "predictive",
            "survey": "evaluation", "discovery": "evaluation"}.get(role, role)

    if workflow == "predictive" and role != "predictive":
        raise RuntimeError(f"Predictive testing requires dataset role 'predictive'; '{dataset_name}' has role '{role}'.")
    if workflow == "evaluation" and role != "evaluation":
        raise RuntimeError(f"Evaluation/discovery requires dataset role 'evaluation'; '{dataset_name}' has role '{role}'.")

    _progress(worker, 15, "Preparing prediction outputs")
    predictor = PredictionManager(threshold_percentile=95.0)
    output_folder = "predictive_test" if workflow == "predictive" else "evaluation"
    save_dir = _dataset_visual_root(cfg, dataset_name) / output_folder

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
            "summary_csv_path": str(Path(save_dir) / "prediction_summary.csv"),
            "metrics_csv_path": str(Path(save_dir) / "prediction_metrics_summary.csv"),
            "metrics": summary.get("metrics", {}),
            "tile_count": summary["tile_count"],
            "dataset": dataset_name,
            "workflow": workflow,
            "model": model_name,
            "show_view": "model",
    }




# ---------------------------------------------------------
# Calibrate one global prediction threshold using ground truth.
# ---------------------------------------------------------
def calibrate_prediction_threshold_task(model_name, dataset_name, model_manager, dataset_manager,
                                        worker=None, percentiles=None, metric="fp_penalised_f1", **kwargs):

    # Reload before calibration so thresholds are selected from the latest model
    # and dataset configuration rather than a stale UI object.
    model_manager.reload()
    dataset_manager.reload()

    _progress(worker, 5, "Checking calibration dataset compatibility")
    compatibility = model_manager.check_dataset_compatibility(model_name, dataset_name)

    if not compatibility.get("compatible"):
        raise RuntimeError(
            "Calibration blocked because the model and dataset profiles do not match.\n"
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

    role = str(getattr(ds_cfg, "role", "mixed") or "mixed").lower()
    role = {"prediction": "predictive", "validation": "predictive", "ground_truth": "predictive"}.get(role, role)
    if role != "predictive":
        raise RuntimeError(f"Threshold calibration requires dataset role 'predictive'; '{dataset_name}' has role '{role}'.")

    # Runtime channel names come from the model's training dataset. They are not
    # duplicated in the model YAML.
    train_ds_name = getattr(cfg, "training_dataset", None)
    train_ds_cfg = dataset_manager.get(train_ds_name) if train_ds_name else None
    if train_ds_cfg is not None:
        cfg.architecture.num_channels = int(getattr(train_ds_cfg, "num_input_channels", 0) or 0)
        cfg.architecture.channel_names = list(getattr(train_ds_cfg, "input_channels", []) or [])

    _progress(worker, 15, "Preparing threshold calibration")
    predictor = PredictionManager(threshold_percentile=95.0)
    save_dir = _dataset_visual_root(cfg, dataset_name) / "calibration"

    if percentiles is None:
        # Keep the old broad sweep, but add more resolution where false-positive
        # controlled thresholds usually land.
        percentiles = [50, 60, 70, 75, 80, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.8, 99.9]

    summary = predictor.calibrate_threshold(model_cfg=cfg,
                                            dataset_cfg=ds_cfg,
                                            save_dir=save_dir,
                                            device=cfg.device,
                                            percentiles=percentiles,
                                            metric=metric,
                                            worker=worker,
    )

    # Save the calibrated threshold into the model config. This does not add any
    # channel duplication; it only records how future predictions should threshold
    # anomaly scores.
    best = summary.get("best", {}) or {}
    prediction_cfg = cfg.cfg.setdefault("prediction", {})
    selection = summary.get("selection", {}) or {}
    prediction_cfg.setdefault("prediction_preset", "Balanced")
    prediction_cfg["threshold_mode"] = "calibrated_global"
    prediction_cfg["threshold_value"] = float(best.get("threshold_value", 0.0) or 0.0)
    prediction_cfg["threshold_percentile"] = float(best.get("threshold_percentile", 0.0) or 0.0)
    prediction_cfg["threshold_metric"] = selection.get("metric", metric or "fp_penalised_f1")
    prediction_cfg["false_positive_penalty"] = float(selection.get("false_positive_penalty", prediction_cfg.get("false_positive_penalty", 0.2)) or 0.2)
    if selection.get("max_false_positive_rate") is not None:
        prediction_cfg["max_false_positive_rate"] = float(selection.get("max_false_positive_rate"))
    if selection.get("min_recall") is not None:
        prediction_cfg["min_recall"] = float(selection.get("min_recall"))
    prediction_cfg["min_component_pixels"] = int(selection.get("min_component_pixels", prediction_cfg.get("min_component_pixels", 0)) or 0)
    prediction_cfg["threshold_calibration_dataset"] = dataset_name
    cfg.save()
    model_manager.reload()

    _progress(worker, 100, "Threshold calibration complete")

    return {"message": "Threshold calibration completed successfully.",
            "output_dir": str(Path(save_dir)),
            "sweep_csv_path": str(Path(save_dir) / "threshold_sweep.csv"),
            "summary_path": str(Path(save_dir) / "calibration_summary.json"),
            "summary_csv_path": str(Path(save_dir) / "calibration_summary.csv"),
            "model_config_path": str(getattr(cfg, "config_path", "")),
            "best": best,
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
    output_dir = _dataset_visual_root(cfg, dataset_name) / "visualisations"
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
