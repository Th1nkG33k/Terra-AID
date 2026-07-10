import csv
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import rasterio

from pathlib import Path
from rasterio.transform import Affine
from PIL import Image
from Core.Pytorch.pytorch_dataset_factory import PyTorchDatasetFactory

# ============================================================
#   PREDICTION MANAGER
#
#   Runs trained Terra-AId reconstruction models over a processed dataset.

#   Output per tile:
#       - anomaly_score.tif       raw floating reconstruction error
#       - anomaly_mask.tif        thresholded anomaly mask
#       - anomaly_heatmap.png     display heatmap
#       - anomaly_overlay.png     heatmap over tile RGB
#       - reconstruction_rgb.png  reconstructed RGB preview

#   Output for the run:
#       - prediction_summary.csv
#       - prediction_summary.json
# ============================================================
class PredictionManager:

    MASK_CANDIDATES = [
                        "ground_truth.tif", "GroundTruth.tif", "GROUND_TRUTH.tif",
                        "labels.tif", "label.tif", "mask.tif", "GT.tif",
    ]

    # Prediction presets are intentionally conservative defaults rather than
    # hard-coded scientific conclusions. They give users repeatable starting
    # points for the precision/recall trade-off while still allowing every
    # value to be overridden in the model configuration.
    PREDICTION_PRESETS = {
        "sensitive": {
            "threshold_metric": "fp_penalised_f1",
            "false_positive_penalty": 0.10,
            "max_false_positive_rate": 0.60,
            "min_recall": 0.30,
            "min_component_pixels": 0,
        },
        "balanced": {
            "threshold_metric": "fp_penalised_f1",
            "false_positive_penalty": 0.20,
            "max_false_positive_rate": 0.45,
            "min_recall": 0.20,
            "min_component_pixels": 0,
        },
        "conservative": {
            "threshold_metric": "fp_penalised_f1",
            "false_positive_penalty": 0.35,
            "max_false_positive_rate": 0.30,
            "min_recall": 0.10,
            "min_component_pixels": 25,
        },
    }

    def __init__(self, threshold_percentile: float = 95.0):
        self.threshold_percentile = float(threshold_percentile)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def predict_dataset(self, model_cfg, dataset_cfg, save_dir=None, device=None, worker=None):

        save_dir = Path(save_dir or (model_cfg.paths.outputs / dataset_cfg.dataset_name / "prediction"))
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = PyTorchDatasetFactory().build(dataset_cfg)

        if len(dataset) == 0:
            raise RuntimeError(f"Dataset '{dataset_cfg.dataset_name}' has no processed tile folders.")

        x0, meta0 = dataset[0]
        dataset_channels = int(x0.shape[0])

        device = self._resolve_device(device or getattr(model_cfg, "device", "cpu"))
        ckpt_path = self._find_checkpoint(model_cfg)

        if worker:
            worker.status(f"Loading checkpoint: {ckpt_path.name}")

        state = torch.load(ckpt_path, map_location=device)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        expected_channels = self._infer_checkpoint_input_channels(state)

        if expected_channels is None:
            expected_channels = int(getattr(model_cfg.architecture, "num_channels", dataset_channels) or dataset_channels)

        # Build the model for the checkpoint depth, not the prediction dataset depth.
        # Prediction tiles are adapted to this depth tile-by-tile below.
        model_cfg.architecture.num_channels = int(expected_channels)
        model = model_cfg.build_model()
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        if worker and dataset_channels != expected_channels:

            worker.status(f"Prediction dataset has {dataset_channels} channels; "
                          f"model checkpoint expects {expected_channels}. Tiles will be padded/truncated consistently."
            )

        expected_channel_names = list(getattr(model_cfg.architecture, "channel_names", []) or [])
        if len(expected_channel_names) != int(expected_channels):
            expected_channel_names = []

        tile_rows = []
        global_scores = []
        run_prediction_cfg = self._prediction_config(model_cfg)

        if worker:
            worker.status(f"Predicting {len(dataset)} tiles from {dataset_cfg.dataset_name}...")

        for idx in range(len(dataset)):
            if worker and worker.cancel_flag:
                break

            x, meta = dataset[idx]
            tile_name = meta.get("tile_id", f"tile_{idx}") if isinstance(meta, dict) else f"tile_{idx}"
            tile_dir = Path(dataset.tile_dirs[idx])
            out_dir = save_dir / tile_name
            out_dir.mkdir(parents=True, exist_ok=True)

            original_channels = int(x.shape[0])
            current_channel_names = meta.get("channel_names", []) if isinstance(meta, dict) else []

            if expected_channel_names and current_channel_names:

                x, channel_action, channel_delta = self._adapt_channels_by_name(x, 
                                                                                current_channel_names, 
                                                                                expected_channel_names
                )
                score_channel_names = expected_channel_names
            else:
                x, channel_action, channel_delta = self._adapt_channels(x, expected_channels)
                score_channel_names = current_channel_names if len(current_channel_names) == int(x.shape[0]) else []

            prediction_cfg = run_prediction_cfg
            threshold_mode = str(prediction_cfg.get("threshold_mode", "per_tile_percentile") or "per_tile_percentile")
            threshold_mode_normalised = threshold_mode.strip().lower()

            score_raw = self._predict_tile_score(model, x, device,
                                                 channel_names=score_channel_names,
                                                 prediction_cfg=prediction_cfg)
            score_raw = self._resize_score_to_source(score_raw, tile_dir)
            score_raw = self._clean_score(score_raw)
            score_norm = self._normalise_score(score_raw)

            # Calibrated thresholds are learned from RAW reconstruction-error scores.
            # Per-tile visual thresholds remain percentile thresholds over the display-normalised score.
            calibrated_modes = {"calibrated_global", "calibrated_global_percentile", "global_calibrated"}
            has_threshold_value = not self._is_missing_config_value(prediction_cfg.get("threshold_value"))

            if threshold_mode_normalised in calibrated_modes and has_threshold_value:
                threshold_mode = "calibrated_global"
                threshold = float(prediction_cfg.get("threshold_value"))
                threshold_score_space = "raw_reconstruction_mse"
                threshold_percentile = prediction_cfg.get("threshold_percentile")
                try:
                    threshold_percentile = float(threshold_percentile) if threshold_percentile not in (None, "") else None
                except (TypeError, ValueError):
                    threshold_percentile = None
                pred_mask = score_raw >= threshold
            else:
                threshold_mode = "per_tile_percentile"
                threshold_score_space = "normalised_display_score"
                threshold = float(np.nanpercentile(score_norm, self.threshold_percentile))
                threshold_percentile = float(self.threshold_percentile)
                pred_mask = score_norm >= threshold

            pred_mask = self._postprocess_mask(pred_mask, prediction_cfg)

            profile = self._source_profile(tile_dir, score_raw.shape)
            self._write_float_tif(out_dir / "anomaly_score.tif", score_raw, profile)
            self._write_float_tif(out_dir / "anomaly_score_display.tif", score_norm, profile)
            self._write_mask_tif(out_dir / "anomaly_mask.tif", pred_mask, profile)

            rgb = self._read_rgb(tile_dir, score_raw.shape)
            heatmap = self._score_to_heatmap(score_norm)
            overlay = self._overlay_heatmap(rgb, heatmap)
            Image.fromarray(heatmap).save(out_dir / "anomaly_heatmap.png")
            Image.fromarray(overlay).save(out_dir / "anomaly_overlay.png")

            recon_rgb = self._reconstruction_rgb(model, x, device)
            Image.fromarray(recon_rgb).save(out_dir / "reconstruction_rgb.png")

            row = {"tile_id": tile_name,
                   "mean_score": float(np.nanmean(score_raw)),
                   "max_score": float(np.nanmax(score_raw)),
                   "mean_display_score": float(np.nanmean(score_norm)),
                   "max_display_score": float(np.nanmax(score_norm)),
                   "threshold": threshold,
                   "threshold_mode": threshold_mode,
                   "threshold_score_space": threshold_score_space,
                   "threshold_percentile": threshold_percentile,
                   "anomaly_pixels": int(pred_mask.sum()),
                   "predicted_pixels": int(pred_mask.sum()),
                   "anomaly_fraction": float(pred_mask.mean()),
                   "input_channels": original_channels,
                   "model_channels": int(expected_channels),
                   "channel_action": channel_action,
                   "channel_delta": int(channel_delta),
                   "has_ground_truth": False,
                   "ground_truth": "",
                   "ground_truth_pixels": 0,
                   "ground_truth_fraction": 0.0,
                   "output_dir": str(out_dir),
            }

            gt_path = self._find_ground_truth(tile_dir)

            if gt_path:

                gt = self._read_mask(gt_path, score_raw.shape)
                row.update(self._mask_metrics(pred_mask, gt))
                row["has_ground_truth"] = True
                row["ground_truth"] = str(gt_path)
                row["ground_truth_pixels"] = int(gt.sum())
                row["ground_truth_fraction"] = float(gt.mean())

            tile_rows.append(row)
            global_scores.append(score_raw.ravel())

            if worker:

                pct = int(((idx + 1) / len(dataset)) * 100)
                worker.progress(f"Prediction {pct}% — {tile_name}")

        summary = {"model": model_cfg.model_name,
                   "dataset": dataset_cfg.dataset_name,
                   "checkpoint": str(ckpt_path),
                   "threshold_percentile": self.threshold_percentile,
                   "score_space": "raw_reconstruction_mse",
                   "display_score_space": "per_tile_percentile_normalised_1_99",
                   "dataset_channels_sample": int(dataset_channels),
                   "model_channels": int(expected_channels),
                   "channel_strategy": "checkpoint_depth_first_then_align_by_channel_name_else_pad_or_truncate",
                   "model_channel_names": expected_channel_names,
                   "prediction_preset": run_prediction_cfg.get("prediction_preset", "Balanced"),
                   "threshold_mode_configured": run_prediction_cfg.get("threshold_mode"),
                   "threshold_value_configured": run_prediction_cfg.get("threshold_value"),
                   "threshold_percentile_configured": run_prediction_cfg.get("threshold_percentile"),
                   "threshold_calibration_dataset": run_prediction_cfg.get("threshold_calibration_dataset"),
                   "threshold_calibration_summary": run_prediction_cfg.get("threshold_calibration_summary"),
                   "threshold_calibration_fallback_used": run_prediction_cfg.get("threshold_calibration_fallback_used"),
                   "threshold_calibration_eligible": run_prediction_cfg.get("threshold_calibration_eligible"),
                   "threshold_metric": run_prediction_cfg.get("threshold_metric"),
                   "false_positive_penalty": run_prediction_cfg.get("false_positive_penalty"),
                   "max_false_positive_rate": run_prediction_cfg.get("max_false_positive_rate"),
                   "min_recall": run_prediction_cfg.get("min_recall"),
                   "min_component_pixels": run_prediction_cfg.get("min_component_pixels"),
                   "tile_count": len(tile_rows),
                   "output_dir": str(save_dir),
                   "tiles": tile_rows,
        }

        if global_scores:
            all_scores = np.concatenate(global_scores)
            summary["global_scores"] = {"mean_score": float(np.nanmean(all_scores)),
                                        "max_score": float(np.nanmax(all_scores)),
                                        "p90_score": float(np.nanpercentile(all_scores, 90)),
                                        "p95_score": float(np.nanpercentile(all_scores, 95)),
                                        "p97_score": float(np.nanpercentile(all_scores, 97)),
                                        "p99_score": float(np.nanpercentile(all_scores, 99)),
                                        "p995_score": float(np.nanpercentile(all_scores, 99.5)),
            }

        summary["metrics"] = self._aggregate_metrics(tile_rows)

        self._write_summary(save_dir, summary, tile_rows)
        return summary


    # ---------------------------------------------------------
    # Threshold calibration
    # ---------------------------------------------------------
    def calibrate_threshold(self, model_cfg, dataset_cfg, save_dir=None, device=None,
                            percentiles=None, metric=None, worker=None):
        """Calibrate one global anomaly threshold using ground-truth masks.

        This is intentionally separate from predict_dataset().  predict_dataset()
        is useful for visual per-tile outputs; calibration asks a different
        question: "what single threshold works best across this labelled
        dataset?"  It therefore collects continuous anomaly scores first, then
        evaluates a sweep of global percentile thresholds against ground_truth.tif.
        """

        save_dir = Path(save_dir or (model_cfg.paths.outputs / dataset_cfg.dataset_name / "calibration"))
        save_dir.mkdir(parents=True, exist_ok=True)

        percentiles = percentiles or [50, 60, 70, 75, 80, 85, 88, 90, 92, 94, 95, 96, 97, 98, 99, 99.5, 99.8, 99.9]
        percentiles = [float(p) for p in percentiles]

        dataset = PyTorchDatasetFactory().build(dataset_cfg)
        if len(dataset) == 0:
            raise RuntimeError(f"Dataset '{dataset_cfg.dataset_name}' has no processed tile folders.")

        x0, meta0 = dataset[0]
        dataset_channels = int(x0.shape[0])

        device = self._resolve_device(device or getattr(model_cfg, "device", "cpu"))
        ckpt_path = self._find_checkpoint(model_cfg)

        if worker:
            worker.status(f"Loading checkpoint for calibration: {ckpt_path.name}")

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        expected_channels = self._infer_checkpoint_input_channels(state)
        if expected_channels is None:
            expected_channels = int(getattr(model_cfg.architecture, "num_channels", dataset_channels) or dataset_channels)

        model_cfg.architecture.num_channels = int(expected_channels)
        model = model_cfg.build_model()
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        expected_channel_names = list(getattr(model_cfg.architecture, "channel_names", []) or [])
        if len(expected_channel_names) != int(expected_channels):
            expected_channel_names = []

        prediction_cfg = self._prediction_config(model_cfg)

        labelled_tiles = []
        all_scores = []
        skipped_tiles = []

        if worker:
            worker.status(f"Collecting anomaly scores from {len(dataset)} tiles...")

        for idx in range(len(dataset)):
            if worker and worker.cancel_flag:
                break

            x, meta = dataset[idx]
            tile_name = meta.get("tile_id", f"tile_{idx}") if isinstance(meta, dict) else f"tile_{idx}"
            tile_dir = Path(dataset.tile_dirs[idx])
            current_channel_names = meta.get("channel_names", []) if isinstance(meta, dict) else []

            if expected_channel_names and current_channel_names:
                x, channel_action, channel_delta = self._adapt_channels_by_name(
                    x, current_channel_names, expected_channel_names
                )
                score_channel_names = expected_channel_names
            else:
                x, channel_action, channel_delta = self._adapt_channels(x, expected_channels)
                score_channel_names = current_channel_names if len(current_channel_names) == int(x.shape[0]) else []

            gt_path = self._find_ground_truth(tile_dir)
            if not gt_path:
                skipped_tiles.append({"tile_id": tile_name, "reason": "no_ground_truth"})
                continue

            score_raw = self._predict_tile_score(model, x, device,
                                                 channel_names=score_channel_names,
                                                 prediction_cfg=prediction_cfg)
            score_raw = self._resize_score_to_source(score_raw, tile_dir)
            score_raw = self._clean_score(score_raw)
            gt = self._read_mask(gt_path, score_raw.shape)

            labelled_tiles.append({
                "tile_id": tile_name,
                "score": score_raw,
                "gt": gt,
                "gt_path": str(gt_path),
                "channel_action": channel_action,
                "channel_delta": int(channel_delta),
            })
            all_scores.append(score_raw.ravel())

            if worker:
                pct = int(((idx + 1) / len(dataset)) * 60)
                worker.progress(f"Calibration {pct}% — collected {tile_name}")

        if not labelled_tiles:
            raise RuntimeError(
                f"No ground_truth.tif files were found in dataset '{dataset_cfg.dataset_name}'. "
                "Run dataset processing/rasterisation before calibration."
            )

        score_values = np.concatenate(all_scores).astype(np.float32)
        sweep_rows = []

        if worker:
            worker.status(f"Sweeping {len(percentiles)} global thresholds...")

        for pctl in percentiles:
            threshold = float(np.nanpercentile(score_values, pctl))
            tp = fp = fn = tn = 0
            predicted_pixels = 0
            ground_truth_pixels = 0
            fp_empty = 0
            pred_empty = 0
            empty_tiles = 0
            positive_tiles = 0

            for item in labelled_tiles:
                pred = item["score"] >= threshold
                pred = self._postprocess_mask(pred, prediction_cfg)
                gt = item["gt"]
                m = self._mask_metrics(pred, gt)
                tp += int(m["tp"])
                fp += int(m["fp"])
                fn += int(m["fn"])
                tn += int(m["tn"])
                pred_count = int(pred.sum())
                gt_count = int(gt.sum())
                predicted_pixels += pred_count
                ground_truth_pixels += gt_count
                if gt_count == 0:
                    empty_tiles += 1
                    fp_empty += int(m["fp"])
                    pred_empty += pred_count
                else:
                    positive_tiles += 1

            metrics = self._metrics_from_counts(tp, fp, fn, tn)
            row = {
                "threshold_percentile": float(pctl),
                "threshold_value": threshold,
                "metric_target": metric,
                "tile_count": len(labelled_tiles),
                "positive_ground_truth_tiles": positive_tiles,
                "empty_ground_truth_tiles": empty_tiles,
                "ground_truth_pixels": ground_truth_pixels,
                "predicted_pixels": predicted_pixels,
                "false_positive_pixels_on_empty_tiles": fp_empty,
                "predicted_pixels_on_empty_tiles": pred_empty,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            row.update(metrics)
            sweep_rows.append(row)

        selection = self._selection_settings(metric, prediction_cfg)

        def _eligible(row):
            max_fpr = selection.get("max_false_positive_rate")
            min_recall = selection.get("min_recall")

            if max_fpr is not None and float(row.get("false_positive_rate", 0.0) or 0.0) > float(max_fpr):
                return False
            if min_recall is not None and float(row.get("recall", 0.0) or 0.0) < float(min_recall):
                return False
            return True

        def _primary_score(row):
            metric_name = selection["metric"]

            if metric_name in row:
                return float(row.get(metric_name, 0.0) or 0.0)

            if metric_name in {"fp_penalised_f1", "fp_penalized_f1"}:
                penalty = float(selection.get("false_positive_penalty", 0.2) or 0.2)
                return float(row.get("f1", 0.0) or 0.0) - penalty * float(row.get("false_positive_rate", 0.0) or 0.0)

            if metric_name in {"precision_with_recall_floor", "precision_recall_floor"}:
                return float(row.get("precision", 0.0) or 0.0)

            return float(row.get("f1", 0.0) or 0.0)

        for row in sweep_rows:
            row["SelectionScore"] = float(_primary_score(row))
            row["EligibleForSelection"] = bool(_eligible(row))
            row["selected"] = False

        eligible_rows = [row for row in sweep_rows if row["EligibleForSelection"]]
        selection_fallback_used = False
        if not eligible_rows:
            eligible_rows = sweep_rows
            selection_fallback_used = True

        def _rank(row):
            primary = float(row.get("SelectionScore", 0.0) or 0.0)
            # Tie-breakers favour overlap, then fewer empty-tile false positives.
            return (primary, float(row.get("iou", 0.0) or 0.0), -float(row.get("false_positive_pixels_on_empty_tiles", 0) or 0))

        best = max(eligible_rows, key=_rank)
        best["selected"] = True

        summary = {
            "model": model_cfg.model_name,
            "dataset": dataset_cfg.dataset_name,
            "checkpoint": str(ckpt_path),
            "threshold_mode": "calibrated_global_percentile",
            "score_space": "raw_reconstruction_mse",
            "prediction_preset": prediction_cfg.get("prediction_preset", "Balanced"),
            "metric_target": selection["metric"],
            "threshold_metric": selection["metric"],
            "false_positive_penalty": selection.get("false_positive_penalty"),
            "max_false_positive_rate": selection.get("max_false_positive_rate"),
            "min_recall": selection.get("min_recall"),
            "min_component_pixels": selection.get("min_component_pixels"),
            "selection_fallback_used": selection_fallback_used,
            "selected_threshold_percentile": best.get("threshold_percentile"),
            "selected_threshold_value": best.get("threshold_value"),
            "selected_selection_score": best.get("SelectionScore"),
            "selected_eligible_for_selection": best.get("EligibleForSelection"),
            "selection": selection,
            "best": best,
            "percentiles_tested": percentiles,
            "labelled_tile_count": len(labelled_tiles),
            "skipped_tiles": skipped_tiles,
            "score_distribution": {
                "mean": float(np.nanmean(score_values)),
                "std": float(np.nanstd(score_values)),
                "min": float(np.nanmin(score_values)),
                "max": float(np.nanmax(score_values)),
                "p90": float(np.nanpercentile(score_values, 90)),
                "p95": float(np.nanpercentile(score_values, 95)),
                "p97": float(np.nanpercentile(score_values, 97)),
                "p99": float(np.nanpercentile(score_values, 99)),
                "p995": float(np.nanpercentile(score_values, 99.5)),
                "p998": float(np.nanpercentile(score_values, 99.8)),
            },
            "output_dir": str(save_dir),
        }

        with open(save_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
            preferred_keys = [
                "threshold_percentile", "threshold_value", "metric_target",
                "SelectionScore", "EligibleForSelection", "selected",
            ]
            keys = preferred_keys + [k for k in sweep_rows[0].keys() if k not in preferred_keys]
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(sweep_rows)

        with open(save_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        calibration_flat = {
            "model": summary.get("model"),
            "dataset": summary.get("dataset"),
            "threshold_mode": summary.get("threshold_mode"),
            "score_space": summary.get("score_space"),
            "prediction_preset": summary.get("prediction_preset"),
            "metric_target": summary.get("metric_target"),
            "false_positive_penalty": summary.get("false_positive_penalty"),
            "max_false_positive_rate": summary.get("max_false_positive_rate"),
            "min_recall": summary.get("min_recall"),
            "min_component_pixels": summary.get("min_component_pixels"),
            "selection_fallback_used": summary.get("selection_fallback_used"),
            "selected_threshold_percentile": summary.get("selected_threshold_percentile"),
            "selected_threshold_value": summary.get("selected_threshold_value"),
            "selected_selection_score": summary.get("selected_selection_score"),
            "selected_eligible_for_selection": summary.get("selected_eligible_for_selection"),
            "selected_precision": best.get("precision"),
            "selected_recall": best.get("recall"),
            "selected_f1": best.get("f1"),
            "selected_iou": best.get("iou"),
            "selected_false_positive_rate": best.get("false_positive_rate"),
            "selected_false_positive_pixels_on_empty_tiles": best.get("false_positive_pixels_on_empty_tiles"),
            "labelled_tile_count": summary.get("labelled_tile_count"),
            "output_dir": summary.get("output_dir"),
        }
        with open(save_dir / "calibration_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(calibration_flat.keys()))
            writer.writeheader()
            writer.writerow(calibration_flat)

        if worker:
            metric_name = selection["metric"]
            worker.status(
                f"Best threshold: p{best['threshold_percentile']} = {best['threshold_value']:.6f} "
                f"({metric_name}: {best.get('SelectionScore', 0):.6f})"
            )
            worker.progress("Calibration 100% — threshold sweep complete")

        return summary

    # ---------------------------------------------------------
    # Model helpers
    # ---------------------------------------------------------
    def _resolve_device(self, device):

        dev = torch.device(device or "cpu")

        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")

        return dev

    def _find_checkpoint(self, model_cfg):

        ckpt_dir = Path(model_cfg.paths.checkpoints)
        preferred = ckpt_dir / f"{model_cfg.model_name}.pt"

        if preferred.exists():
            return preferred
        
        matches = sorted(ckpt_dir.glob("*.pt"))
        
        if not matches:
            raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")
        
        return matches[0]


    # ----------------------------------------------------------------------------
        # Infer the channel depth the checkpoint was trained with directly from
        # the saved weights. This is more reliable than dataset_cfg.depth because
        # DEM/SOIL/indices/QC may be added or missing after download.
    # ----------------------------------------------------------------------------
    def _infer_checkpoint_input_channels(self, state):

        if not isinstance(state, dict):
            return None

        candidate_suffixes = ["input_projection.weight",          # ResNet autoencoder
                              "encoder.encoder.0.block.0.weight", # MAE encoder
                              "encoder_blocks.0.block.0.weight",  # Classic CNN AE
        ]

        for suffix in candidate_suffixes:

            for key, value in state.items():
            
                if key.endswith(suffix) and torch.is_tensor(value) and value.ndim >= 2:
                    return int(value.shape[1])

        # Fallback: first 2D+ conv-like weight where dim1 looks like a realistic
        # input channel count and dim0 is not itself the reconstructed channel count.
        for key, value in state.items():

            if torch.is_tensor(value) and value.ndim == 4 and 1 <= int(value.shape[1]) <= 128:
                return int(value.shape[1])
            
        return None


# ----------------------------------------------------------------------------
        # Make a prediction tile match the trained model depth.

        # - If the prediction tile has fewer channels, append zero-filled channels.
        # - If it has more channels, truncate extras from the end.

        # The dataset loader always orders channels as S2 -> indices -> DEM -> SOIL -> QC,
        # so appending zeros is stable for the common case where optional layers are
        # missing from the prediction dataset.
# ----------------------------------------------------------------------------
    def _adapt_channels(self, x, expected_channels):

        current_channels = int(x.shape[0])
        expected_channels = int(expected_channels)

        if current_channels == expected_channels:
            return x, "none", 0

        if current_channels < expected_channels:

            pad_count = expected_channels - current_channels
            pad = torch.zeros((pad_count, x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0), "padded", pad_count
        
        return x[:expected_channels], "truncated", current_channels - expected_channels


# ----------------------------------------------------------------------------
        # Align a prediction tile to the model's training channels by name.
        # Missing channels are zero-filled; extra channels are ignored.
# ----------------------------------------------------------------------------
    def _adapt_channels_by_name(self, x, current_names, expected_names):

        current_lookup = {str(name): i for i, name in enumerate(current_names)}
        aligned = []
        missing = []

        for expected_name in expected_names:

            expected_name = str(expected_name)
            idx = current_lookup.get(expected_name)
            
            if idx is None or idx >= int(x.shape[0]):
                aligned.append(torch.zeros((1, x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device))
                missing.append(expected_name)
            
            else:
                aligned.append(x[idx:idx+1])
        
        if not aligned:
            return self._adapt_channels(x, len(expected_names))
        
        y = torch.cat(aligned, dim=0)
        extra = max(0, int(x.shape[0]) - len(set(current_lookup).intersection(set(map(str, expected_names)))))
        
        if missing and extra:
            return y, "name_aligned_missing_and_extra", len(missing) + extra
        
        if missing:
            return y, "name_aligned_missing_padded", len(missing)
        
        if extra:
            return y, "name_aligned_extra_ignored", extra
        
        return y, "name_aligned", 0

    def _unpack_reconstruction(self, output):

        if isinstance(output, tuple):
            return output[0]
        
        if isinstance(output, dict) and "reconstruction" in output:
            return output["reconstruction"]
        
        return output

    def _is_missing_config_value(self, value):

        if value is None:
            return True

        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
            return True

        return False

    def _prediction_config(self, model_cfg):

        if hasattr(model_cfg, "cfg") and isinstance(getattr(model_cfg, "cfg", None), dict):
            raw = dict(model_cfg.cfg.get("prediction", {}) or {})
        else:
            raw = {}

        raw_preset = raw.get("prediction_preset", raw.get("preset", "Balanced"))
        preset_label = "Balanced" if self._is_missing_config_value(raw_preset) else str(raw_preset).strip()
        preset_name = preset_label.lower()
        preset = dict(self.PREDICTION_PRESETS.get(preset_name, self.PREDICTION_PRESETS["balanced"]))

        # Explicit user/model values override the preset. Empty/null-like values
        # are treated as missing so the chosen preset remains active. This prevents
        # old YAML values such as 'null' or 'None' from disabling the guardrails.
        for key, value in raw.items():
            if not self._is_missing_config_value(value):
                preset[key] = value

        preset["prediction_preset"] = preset_label
        return preset

    def _as_list(self, value):

        if value is None:
            return []

        if isinstance(value, (list, tuple, set)):
            return [str(v).strip() for v in value if str(v).strip()]

        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]

        return [str(value).strip()] if str(value).strip() else []

    def _selection_settings(self, metric, prediction_cfg):

        metric_name = str(metric or prediction_cfg.get("threshold_metric") or "fp_penalised_f1").lower()

        def _optional_float(key):
            value = prediction_cfg.get(key)
            if self._is_missing_config_value(value):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return {
            "metric": metric_name,
            "false_positive_penalty": float(prediction_cfg.get("false_positive_penalty", 0.2) or 0.2),
            "max_false_positive_rate": _optional_float("max_false_positive_rate"),
            "min_recall": _optional_float("min_recall"),
            "min_component_pixels": int(prediction_cfg.get("min_component_pixels", 0) or 0),
        }

    def _model_prediction_output(self, model, xb):

        predict_fn = getattr(model, "predict", None)

        if callable(predict_fn):
            return predict_fn(xb)

        return model(xb)

    def _score_channel_indices(self, channel_names, num_channels, prediction_cfg):

        num_channels = int(num_channels)
        names = [str(n) for n in (channel_names or [])]

        if len(names) != num_channels:
            return list(range(num_channels)), []

        include = self._as_list(prediction_cfg.get("score_channels") or prediction_cfg.get("anomaly_score_channels"))
        if include:
            include_upper = {name.upper() for name in include}
            indices = [i for i, name in enumerate(names) if name.upper() in include_upper]
            return (indices or list(range(num_channels))), include

        ignore = self._as_list(prediction_cfg.get("ignore_score_channels"))
        if not ignore:
            ignore = ["SCL", "QC", "MASK", "GROUND_TRUTH", "LABEL"]

        ignore_upper = {name.upper() for name in ignore}
        indices = [i for i, name in enumerate(names) if name.upper() not in ignore_upper]

        return (indices or list(range(num_channels))), [names[i] for i in indices]

    def _predict_tile_score(self, model, x, device, channel_names=None, prediction_cfg=None):

        prediction_cfg = prediction_cfg or {}

        with torch.no_grad():
            xb = x.unsqueeze(0).to(device)
            out = self._model_prediction_output(model, xb)
            recon = self._unpack_reconstruction(out)
            
            if recon.shape[-2:] != xb.shape[-2:]:
                recon = F.interpolate(recon, size=xb.shape[-2:], mode="bilinear", align_corners=False)

            err_channels = (recon - xb).pow(2).squeeze(0)
            score_indices, _ = self._score_channel_indices(channel_names, err_channels.shape[0], prediction_cfg)
            err = err_channels[score_indices].mean(dim=0)
        
        return err.detach().cpu().numpy().astype(np.float32)

    def _reconstruction_rgb(self, model, x, device):

        with torch.no_grad():
            xb = x.unsqueeze(0).to(device)
            recon = self._unpack_reconstruction(self._model_prediction_output(model, xb)).squeeze(0).detach().cpu()
        
        return self._tensor_rgb_uint8(recon)

    def _postprocess_mask(self, mask, prediction_cfg):

        mask = np.asarray(mask).astype(bool)
        min_pixels = int((prediction_cfg or {}).get("min_component_pixels", 0) or 0)

        if min_pixels <= 1 or mask.size == 0:
            return mask

        return self._remove_small_components(mask, min_pixels=min_pixels)

    def _remove_small_components(self, mask, min_pixels=25):

        mask = np.asarray(mask).astype(bool)
        h, w = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        out = np.zeros(mask.shape, dtype=bool)
        neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]

        for y in range(h):
            for x in range(w):
                if visited[y, x] or not mask[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                component = []

                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))

                    for dy, dx in neighbours:
                        ny, nx = cy + dy, cx + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))

                if len(component) >= int(min_pixels):
                    ys, xs = zip(*component)
                    out[list(ys), list(xs)] = True

        return out

    # ---------------------------------------------------------
    # Raster/image helpers
    # ---------------------------------------------------------
    def _find_case_insensitive(self, tile_dir: Path, filename: str):

        target = filename.lower()
        
        for f in tile_dir.iterdir():
            if f.name.lower() == target:
                return f
        
        return None

    def _source_raster_path(self, tile_dir: Path):

        for name in ["RGB.tif", "S2_stack.tif", "indices.tif", "DEM.tif"]:
        
            p = self._find_case_insensitive(tile_dir, name)
        
            if p:
                return p
        
        return None

    def _source_profile(self, tile_dir: Path, shape):

        src_path = self._source_raster_path(tile_dir)
        
        if src_path:
        
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
            profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
        
            return profile

        h, w = shape

        return {"driver": "GTiff", "height": h, "width": w, "count": 1,
                "dtype": "float32", "transform": Affine.identity(), "crs": None,
                "compress": "deflate",
        }

    def _resize_score_to_source(self, score, tile_dir: Path):

        src_path = self._source_raster_path(tile_dir)
        
        if not src_path:
            return score
        
        with rasterio.open(src_path) as src:
            target_shape = (src.height, src.width)
        
        if score.shape == target_shape:
            return score
        
        img = Image.fromarray(score.astype(np.float32), mode="F")
        img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        
        return np.array(img, dtype=np.float32)

    def _clean_score(self, score):

        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _normalise_score(self, score):
        
        score = np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.nanpercentile(score, (1, 99))
        
        if hi - lo <= 1e-8:
            return np.zeros_like(score, dtype=np.float32)
        
        return np.clip((score - lo) / (hi - lo), 0, 1).astype(np.float32)

    def _write_float_tif(self, path, arr, profile):
        
        profile = profile.copy()
        profile.update(count=1, dtype="float32")
        
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)

    def _write_mask_tif(self, path, mask, profile):
        
        profile = profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0)
        
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

    def _read_rgb(self, tile_dir: Path, target_shape):
        
        rgb_path = self._find_case_insensitive(tile_dir, "RGB.tif")
        
        if rgb_path:
        
            with rasterio.open(rgb_path) as src:
                arr = src.read([1, 2, 3]).transpose(1, 2, 0)
        
            if arr.shape[:2] != target_shape:
                img = Image.fromarray(self._normalise_rgb(arr))
                img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
                return np.array(img)
        
            return self._normalise_rgb(arr)
        
        return np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)

    def _normalise_rgb(self, rgb):
        
        rgb = rgb.astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.float32)
        
        for c in range(3):
            band = rgb[:, :, c]
            lo, hi = np.nanpercentile(band, (2, 98))
        
            if hi - lo > 1e-6:
                out[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
        
        return (out * 255).astype(np.uint8)

    def _tensor_rgb_uint8(self, x):
        
        arr = x.detach().cpu().numpy()
        
        if arr.shape[0] >= 4:
            rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)
        
        elif arr.shape[0] >= 3:
            rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
        
        else:
            rgb = np.repeat(arr[0][..., None], 3, axis=-1)
        
        rgb = np.clip(rgb, 0, 1)
        
        return (rgb * 255).astype(np.uint8)

    def _score_to_heatmap(self, score):
        
        # Lightweight red/yellow heatmap without requiring OpenCV.
        s = np.clip(score, 0, 1)
        r = (255 * s).astype(np.uint8)
        g = (255 * np.clip((s - 0.35) / 0.65, 0, 1)).astype(np.uint8)
        b = (80 * (1 - s)).astype(np.uint8)
        
        return np.stack([r, g, b], axis=-1)

    def _overlay_heatmap(self, rgb, heatmap, alpha=0.45):
        
        out = (1 - alpha) * rgb.astype(np.float32) + alpha * heatmap.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # Ground truth metrics
    # ---------------------------------------------------------
    def _find_ground_truth(self, tile_dir: Path):
        
        for name in self.MASK_CANDIDATES:
            p = self._find_case_insensitive(tile_dir, name)
        
            if p:
                return p
        
        return None

    def _read_mask(self, path: Path, target_shape):
        
        with rasterio.open(path) as src:
            arr = src.read(1)
        
        mask = np.nan_to_num(arr, nan=0.0) > 0
        
        if mask.shape != target_shape:
            img = Image.fromarray(mask.astype(np.uint8) * 255)
            img = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            mask = np.array(img) > 0
        
        return mask

    def _mask_metrics(self, pred, gt):
        """Compute binary mask metrics using Python numeric types.

        Raster masks can contain hundreds of thousands or millions of pixels.
        The MCC denominator multiplies several large count values; passing that
        large Python integer through np.sqrt can produce an object-dtype ufunc
        error on Windows/NumPy.  Force all counts to plain Python ints and use
        math.sqrt on a float denominator.
        """

        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)

        if pred.shape != gt.shape:
            raise RuntimeError(
                f"Prediction mask and ground-truth mask shapes do not match: "
                f"pred={pred.shape}, gt={gt.shape}"
            )

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        total = int(tp + fp + fn + tn)

        metrics = self._metrics_from_counts(tp, fp, fn, tn)
        metrics.update({
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_pixels": total,
        })
        return metrics

    def _aggregate_metrics(self, rows):
        """Return global/micro and macro metrics over all tiles with ground truth."""
        gt_rows = [r for r in rows if r.get("has_ground_truth")]

        result = {"tiles_with_ground_truth": len(gt_rows),
                  "tiles_without_ground_truth": len(rows) - len(gt_rows),
                  "tile_count": len(rows),
        }

        if not gt_rows:
            result["available"] = False
            return result

        tp = int(sum(int(r.get("tp", 0) or 0) for r in gt_rows))
        fp = int(sum(int(r.get("fp", 0) or 0) for r in gt_rows))
        fn = int(sum(int(r.get("fn", 0) or 0) for r in gt_rows))
        tn = int(sum(int(r.get("tn", 0) or 0) for r in gt_rows))

        micro = self._metrics_from_counts(tp, fp, fn, tn)
        result.update({"available": True,
                       "micro": micro,
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "ground_truth_pixels": int(sum(int(r.get("ground_truth_pixels", 0) or 0) for r in gt_rows)),
                       "predicted_pixels": int(sum(int(r.get("predicted_pixels", r.get("anomaly_pixels", 0)) or 0) for r in gt_rows)),
        })

        metric_names = ["precision", "recall", "specificity", "false_positive_rate",
                        "false_negative_rate", "f1", "dice", "iou", "accuracy",
                        "balanced_accuracy", "mcc"]
        result["macro"] = {}
        for name in metric_names:
            vals = [float(r[name]) for r in gt_rows if name in r and r[name] not in (None, "")]
            result["macro"][name] = float(np.mean(vals)) if vals else 0.0

        empty_tiles = [r for r in gt_rows if int(r.get("ground_truth_pixels", 0) or 0) == 0]
        positive_tiles = [r for r in gt_rows if int(r.get("ground_truth_pixels", 0) or 0) > 0]
        result["empty_ground_truth_tiles"] = len(empty_tiles)
        result["positive_ground_truth_tiles"] = len(positive_tiles)
        result["false_positive_pixels_on_empty_tiles"] = int(sum(int(r.get("fp", 0) or 0) for r in empty_tiles))
        result["predicted_pixels_on_empty_tiles"] = int(sum(int(r.get("predicted_pixels", r.get("anomaly_pixels", 0)) or 0) for r in empty_tiles))

        return result

    def _metrics_from_counts(self, tp, fp, fn, tn):
        """Compute metrics from TP/FP/FN/TN counts safely."""

        tp = int(tp or 0)
        fp = int(fp or 0)
        fn = int(fn or 0)
        tn = int(tn or 0)
        total = int(tp + fp + fn + tn)

        def safe_div(a, b):
            return float(a) / float(b) if b else 0.0

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        fpr = safe_div(fp, fp + tn)
        fnr = safe_div(fn, fn + tp)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        iou = safe_div(tp, tp + fp + fn)
        accuracy = safe_div(tp + tn, total)
        balanced_accuracy = 0.5 * (recall + specificity)

        # Matthews correlation coefficient.  Do not use np.sqrt here: the
        # denominator can be a very large Python int, which NumPy may treat as
        # object and fail with: AttributeError: 'int' object has no attribute 'sqrt'.
        mcc_product = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_den = math.sqrt(mcc_product) if mcc_product > 0.0 else 0.0
        mcc = safe_div((tp * tn) - (fp * fn), mcc_den)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "f1": float(f1),
            "dice": float(f1),
            "iou": float(iou),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "mcc": float(mcc),
            "total_pixels": int(total),
        }

    def _write_summary(self, save_dir: Path, summary: dict, rows: list[dict]):
        
        with open(save_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if not rows:
            return
        
        preferred = ["tile_id", "has_ground_truth", "ground_truth_pixels", "predicted_pixels",
                     "tp", "fp", "fn", "tn", "precision", "recall", "specificity",
                     "f1", "dice", "iou", "accuracy", "balanced_accuracy", "mcc",
                     "false_positive_rate", "false_negative_rate", "anomaly_fraction",
                     "mean_score", "max_score", "mean_display_score", "max_display_score",
                     "threshold", "threshold_mode", "threshold_score_space",
                     "threshold_percentile", "input_channels", "model_channels",
                     "channel_action", "channel_delta", "ground_truth", "output_dir"]
        keys = preferred + sorted({k for row in rows for k in row.keys()} - set(preferred))
        
        with open(save_dir / "prediction_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        # Flat one-row metrics file for easy comparison between model runs.
        metrics = summary.get("metrics", {}) or {}
        if metrics.get("available"):
            flat = {"model": summary.get("model"),
                    "dataset": summary.get("dataset"),
                    "checkpoint": summary.get("checkpoint"),
                    "prediction_preset": summary.get("prediction_preset"),
                    "threshold_mode_configured": summary.get("threshold_mode_configured"),
                    "threshold_value_configured": summary.get("threshold_value_configured"),
                    "threshold_percentile_configured": summary.get("threshold_percentile_configured"),
                    "threshold_calibration_dataset": summary.get("threshold_calibration_dataset"),
                    "threshold_calibration_summary": summary.get("threshold_calibration_summary"),
                    "threshold_calibration_fallback_used": summary.get("threshold_calibration_fallback_used"),
                    "threshold_calibration_eligible": summary.get("threshold_calibration_eligible"),
                    "threshold_metric": summary.get("threshold_metric"),
                    "false_positive_penalty": summary.get("false_positive_penalty"),
                    "max_false_positive_rate": summary.get("max_false_positive_rate"),
                    "min_recall": summary.get("min_recall"),
                    "min_component_pixels": summary.get("min_component_pixels"),
                    "tile_count": metrics.get("tile_count"),
                    "tiles_with_ground_truth": metrics.get("tiles_with_ground_truth"),
                    "positive_ground_truth_tiles": metrics.get("positive_ground_truth_tiles"),
                    "empty_ground_truth_tiles": metrics.get("empty_ground_truth_tiles"),
                    "ground_truth_pixels": metrics.get("ground_truth_pixels"),
                    "predicted_pixels": metrics.get("predicted_pixels"),
                    "tp": metrics.get("tp"), "fp": metrics.get("fp"),
                    "fn": metrics.get("fn"), "tn": metrics.get("tn"),
                    "false_positive_pixels_on_empty_tiles": metrics.get("false_positive_pixels_on_empty_tiles"),
                    "predicted_pixels_on_empty_tiles": metrics.get("predicted_pixels_on_empty_tiles"),
            }
            for prefix in ("micro", "macro"):
                for k, v in (metrics.get(prefix, {}) or {}).items():
                    flat[f"{prefix}_{k}"] = v

            with open(save_dir / "prediction_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
                keys = list(flat.keys())
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerow(flat)
