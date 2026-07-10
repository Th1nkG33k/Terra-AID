import json
from pathlib import Path

import PySimpleGUI as sg

from Interface.theme import COLORS, RText, RButton
from Interface.pages.view_base import ViewerBase
from Interface.controls.control_select_dataset import ControlSelectDataset
from Core.Utils.image_utility import ImageUtility


class ModelViewer(ViewerBase):
    key = "-PAGE_VIEWER_MODEL-"

    def __init__(self, dataset_manager=None, model_manager=None):
        super().__init__(entity=None, title="Model Viewer")

        self.dataset_manager = dataset_manager
        self.model_manager = model_manager
        self.model_cfg = None
        self.cfg = None
        self.window = None
        self.img_util = ImageUtility()

        # These are UI selections only. Dataset purpose is defined by the
        # dataset role, not by writing dataset mappings into the model YAML.
        self.selected_predictive_dataset = None   # role: predictive, labelled model test/calibration data
        self.selected_evaluation_dataset = None   # role: evaluation, unlabelled anomaly discovery data

        self.txt_name = None
        self.txt_arch = None
        self.txt_optimizer = None
        self.txt_training = None
        self.txt_scheduler = None
        self.txt_profile = None
        self.txt_dataset = None
        self.txt_predictive = None
        self.txt_evaluation = None
        self.txt_profile_status = None

    # ------------------------------------------------------------
    # Build stacked views (static layout)
    # ------------------------------------------------------------
    def build_views(self):
        self.txt_name = RText("Name: -")
        self.txt_arch = RText("Architecture: -")
        self.txt_optimizer = RText("Optimizer: -")
        self.txt_training = RText("Training: -")
        self.txt_scheduler = RText("Scheduler: -")
        self.txt_profile = RText("Input Profile: -")
        self.txt_dataset = RText("Training Dataset: -")
        self.txt_predictive = RText("Predictive/Test Dataset: -")
        self.txt_evaluation = RText("Evaluation/Discovery Dataset: -")
        self.txt_profile_status = RText("Profile Check: -")

        info_layout = [
            [RText("Information")],
            [self.txt_name],
            [self.txt_arch],
            [self.txt_optimizer],
            [self.txt_training],
            [self.txt_scheduler],
            [self.txt_profile],
            [self.txt_dataset],
        ]

        view_info = sg.Column(info_layout, key=f"{self.key}_INFO", visible=False,
                              background_color=COLORS["bg_dark"])
        self.add_view("info", view_info)

        vis_layout = [
            [RText("Visualisations")],
            [
                RButton("Results", key=f"{self.key}_RESULTS"),
                RButton("Loss Curve", key=f"{self.key}_LOSS_CURVE"),
                RButton("Loss Metrics", key=f"{self.key}_LOSS_METRICS"),
                RButton("Heatmap", key=f"{self.key}_HEATMAP"),
                RButton("Arch Score", key=f"{self.key}_ARCH_SCORE"),
                RButton("Anomaly Map", key=f"{self.key}_ANOMALY_MAP"),
                RButton("Clustering", key=f"{self.key}_CLUSTER"),
            ],
        ]

        view_visuals = sg.Column(vis_layout, key=f"{self.key}_VISUALS", visible=False,
                                 background_color=COLORS["bg_dark"])
        self.add_view("visuals", view_visuals)

        data_layout = [
            [RText("Dataset")],
            [RButton("Select Training Dataset", key=f"{self.key}_TRAIN_DATASET")],
        ]

        view_dataset = sg.Column(data_layout, key=f"{self.key}_TRAINING_DATASET", visible=False,
                                 background_color=COLORS["bg_dark"])
        self.add_view("dataset", view_dataset)

        train_layout = [
            [RText("Training")],
            [RButton("Train", key=f"{self.key}_TRAIN")],
        ]

        view_training = sg.Column(train_layout, key=f"{self.key}_TRAINING", visible=False,
                                  background_color=COLORS["bg_dark"])
        self.add_view("training", view_training)

        prediction_layout = [
            [RText("Predictive Testing / Calibration")],
            [self.txt_predictive],
            [
                RButton("Select Predictive Dataset", key=f"{self.key}_PREDICTIVE_DATASET"),
                RButton("Predict + Metrics", key=f"{self.key}_PREDICTIVE_TEST"),
                RButton("Calibrate Threshold", key=f"{self.key}_CALIBRATE_THRESHOLD"),
            ],
            [RText("Evaluation / Discovery")],
            [self.txt_evaluation],
            [
                RButton("Select Evaluation Dataset", key=f"{self.key}_EVALUATION_DATASET"),
                RButton("Find Anomalies", key=f"{self.key}_EVALUATE"),
            ],
            [self.txt_profile_status],
        ]

        view_prediction = sg.Column(prediction_layout, key=f"{self.key}_PREDICTION", visible=False,
                                    background_color=COLORS["bg_dark"])
        self.add_view("prediction", view_prediction)

    # ------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------
    def _safe(self, value, default="-"):
        return default if value in (None, "") else value

    def _channels_text(self, channels):
        channels = list(channels or [])
        return ", ".join(map(str, channels)) if channels else "-"

    def _training_dataset_channels(self, cfg):
        if not self.dataset_manager or not getattr(cfg, "training_dataset", None):
            return []
        ds = self.dataset_manager.get(cfg.training_dataset)
        return list(getattr(ds, "input_channels", []) or []) if ds else []

    def _architecture_summary(self, cfg):
        arc = getattr(cfg, "architecture", None)
        if arc is None:
            return "Architecture: -"

        arc_type = str(getattr(arc, "type", "mae"))
        derived_channels = self._training_dataset_channels(cfg)
        channel_count = len(derived_channels) if derived_channels else getattr(arc, "num_channels", None)
        parts = [f"Architecture: {arc_type}", f"Channels: {self._safe(channel_count)}"]

        if arc_type == "mae":
            parts.extend([
                f"Enc Depth: {self._safe(getattr(arc, 'encoder_depth', None))}",
                f"Dec Depth: {self._safe(getattr(arc, 'decoder_depth', None))}",
                f"Embed Dim: {self._safe(getattr(arc, 'embed_dim', None))}",
                f"Dec Dim: {self._safe(getattr(arc, 'decoder_dim', None))}",
                f"Mask Ratio: {self._safe(getattr(arc, 'mask_ratio', None))}",
                f"Base Channels: {self._safe(getattr(arc, 'base_channels', None))}",
            ])
        elif arc_type == "cnn_autoencoder":
            parts.extend([
                f"Depth: {self._safe(getattr(arc, 'encoder_depth', None))}",
                f"Base Channels: {self._safe(getattr(arc, 'base_channels', None))}",
                f"Latent Channels: {self._safe(getattr(arc, 'latent_channels', None))}",
            ])
        elif arc_type == "resnet_autoencoder":
            parts.extend([
                f"Backbone: {self._safe(getattr(arc, 'backbone', None))}",
                f"Pretrained: {self._safe(getattr(arc, 'pretrained', None))}",
                f"Freeze Epochs: {self._safe(getattr(arc, 'freeze_encoder_epochs', None))}",
            ])

        return "  |  ".join(parts)

    # ------------------------------------------------------------
    # Load model dynamically (called from mainW)
    # ------------------------------------------------------------
    def load_model(self, cfg, window):
        self.model_cfg = cfg
        self.cfg = cfg
        self.window = window

        self.txt_name.update(f"Name: {cfg.model_name}  |  Stage: {cfg.stage}")
        self.txt_arch.update(self._architecture_summary(cfg))

        opt = cfg.optimizer
        self.txt_optimizer.update(
            f"Optimizer: {self._safe(opt.type)}  |  Weight Decay: {self._safe(opt.weight_decay)}  |  lr: {self._safe(opt.lr)}"
        )

        tr = cfg.training
        self.txt_training.update(
            f"Batch Size: {self._safe(tr.batch_size)}  |  Num Workers: {self._safe(tr.num_workers)}  |  "
            f"Device: {cfg.device}  |  Epochs: {self._safe(tr.epochs)}  |  "
            f"Early Stop Patience: {self._safe(tr.early_stopping_patience)}"
        )

        sch = cfg.scheduler
        self.txt_scheduler.update(f"Scheduler: {self._safe(sch.type)}  |  Warmup Epochs: {self._safe(sch.warmup_epochs)}")

        derived_channels = self._training_dataset_channels(cfg)
        if derived_channels:
            self.txt_profile.update(
                f"Derived Model Inputs: {len(derived_channels)} channels  |  {self._channels_text(derived_channels)}"
            )
        else:
            self.txt_profile.update("Derived Model Inputs: choose a training dataset")

        self.txt_dataset.update(f"Training Dataset: {self._safe(getattr(cfg, 'training_dataset', None))}")
        self.txt_predictive.update(f"Predictive/Test Dataset: {self._safe(self.selected_predictive_dataset)}")
        self.txt_evaluation.update(f"Evaluation/Discovery Dataset: {self._safe(self.selected_evaluation_dataset)}")
        self.txt_profile_status.update("Profile Check: -")

    # ------------------------------------------------------------
    # Event handler for viewer buttons
    # ------------------------------------------------------------
    def handle_event(self, event, values, window):
        handled = True

        if event == f"{self.key}_RESULTS":
            self.show_visualisation("Results", window)
        elif event == f"{self.key}_LOSS_CURVE":
            self.show_visualisation("Loss Curve", window)
        elif event == f"{self.key}_LOSS_METRICS":
            self.show_visualisation("Loss Metrics", window)
        elif event == f"{self.key}_HEATMAP":
            self.show_visualisation("Heatmap", window)
        elif event == f"{self.key}_ARCH_SCORE":
            self.show_visualisation("Architecture Score", window)
        elif event == f"{self.key}_ANOMALY_MAP":
            self.start_model_visual_task(window, "anomaly_map")
        elif event == f"{self.key}_CLUSTER":
            self.start_model_visual_task(window, "clustering")
        elif event == f"{self.key}_TRAIN":
            self.start_training(window)
        elif event == f"{self.key}_TRAIN_DATASET":
            self.select_training_dataset(window)
        elif event == f"{self.key}_PREDICTIVE_DATASET":
            self.select_predictive_dataset(window)
        elif event == f"{self.key}_PREDICTIVE_TEST":
            self.start_predictive_test(window)
        elif event == f"{self.key}_CALIBRATE_THRESHOLD":
            self.start_threshold_calibration(window)
        elif event == f"{self.key}_EVALUATION_DATASET":
            self.select_evaluation_dataset(window)
        elif event == f"{self.key}_EVALUATE":
            self.start_evaluation(window)
        else:
            handled = False

        return handled

    # ------------------------------------------------------------
    # Dataset selection helpers
    # ------------------------------------------------------------
    def select_training_dataset(self, window):
        if not self._require_model():
            return

        selector = ControlSelectDataset(self.dataset_manager, mode="training")
        selected_dataset = selector.show(window)

        if selected_dataset:
            window.write_event_value(
                "-TASK_MODEL_SET_DATASET-",
                {"model_name": self.cfg.model_name, "dataset_name": selected_dataset},
            )

    def select_predictive_dataset(self, window):
        if not self._require_model():
            return

        selector = ControlSelectDataset(self.dataset_manager, mode="predictive")
        selected_dataset = selector.show(window)

        if selected_dataset:
            self.selected_predictive_dataset = selected_dataset
            self.txt_predictive.update(f"Predictive/Test Dataset: {selected_dataset}")
            compatibility = self.model_manager.check_dataset_compatibility(self.cfg.model_name, selected_dataset) if self.model_manager else None
            self.update_profile_status(compatibility)

    def select_evaluation_dataset(self, window):
        if not self._require_model():
            return

        selector = ControlSelectDataset(self.dataset_manager, mode="evaluation")
        selected_dataset = selector.show(window)

        if selected_dataset:
            self.selected_evaluation_dataset = selected_dataset
            self.txt_evaluation.update(f"Evaluation/Discovery Dataset: {selected_dataset}")
            compatibility = self.model_manager.check_dataset_compatibility(self.cfg.model_name, selected_dataset) if self.model_manager else None
            self.update_profile_status(compatibility)

    # ------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------
    def _require_model(self):
        if not self.cfg:
            sg.popup_error("No model is currently loaded.")
            return False
        return True

    def _training_log_path(self) -> Path:
        return Path(self.cfg.paths.logs) / "training_log.json"

    def _visual_path(self, filename: str) -> Path:
        dataset_name = getattr(self.cfg, "training_dataset", None) or "training"
        out_dir = Path(self.cfg.paths.outputs) / str(dataset_name) / "visualisations"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename

    def _load_training_log(self):
        log_path = self._training_log_path()
        if not log_path.exists():
            raise FileNotFoundError(f"Training log not found:\n{log_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)

        if not log:
            raise ValueError(f"Training log is empty:\n{log_path}")

        return log

    def _channel_labels(self, count: int):
        labels = self._training_dataset_channels(self.cfg)
        if len(labels) == count:
            return labels

        arc_labels = list(getattr(getattr(self.cfg, "architecture", None), "channel_names", []) or [])
        if len(arc_labels) == count:
            return arc_labels

        fallback = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "BSI", "DEM", "SOIL", "QC"]
        if len(fallback) >= count:
            return fallback[:count]

        return [f"Ch {i}" for i in range(count)]

    def _show_results_summary(self, log):
        last = log[-1]
        best = min(log, key=lambda e: e.get("val", {}).get("mse", float("inf")))

        body = (
            f"Model: {self.cfg.model_name}\n"
            f"Stage: {self.cfg.stage}\n"
            f"Training Dataset: {getattr(self.cfg, 'training_dataset', None)}\n\n"
            f"Epochs logged: {len(log)}\n\n"
            f"Last epoch: {last.get('epoch')}\n"
            f"  Train MSE: {last.get('train', {}).get('mse')}\n"
            f"  Train MAE: {last.get('train', {}).get('mae')}\n"
            f"  Train PSNR: {last.get('train', {}).get('psnr')}\n"
            f"  Val MSE: {last.get('val', {}).get('mse')}\n"
            f"  Val MAE: {last.get('val', {}).get('mae')}\n"
            f"  Val PSNR: {last.get('val', {}).get('psnr')}\n\n"
            f"Best validation epoch: {best.get('epoch')}\n"
            f"  Best Val MSE: {best.get('val', {}).get('mse')}\n"
        )

        sg.popup_scrolled(body, title="Model Results", size=(80, 24))

    def show_visualisation(self, vis_type, window):
        if not self._require_model():
            return

        print(f"[ModelViewer] Showing visualisation: {vis_type}")

        try:
            if vis_type == "Results":
                log = self._load_training_log()
                self._show_results_summary(log)
                return

            if vis_type == "Loss Curve":
                log = self._load_training_log()
                out = self._visual_path("loss_curve.png")
                self.img_util.plot_loss_curve(log, save_path=out)
                self.img_util.show_image_window(out, title="Loss Curve")
                return

            if vis_type == "Loss Metrics":
                log = self._load_training_log()
                out = self._visual_path("loss_metrics.png")
                self.img_util.plot_training_metrics(log, save_path=out)
                self.img_util.show_image_window(out, title="Loss Metrics")
                return

            if vis_type == "Heatmap":
                log = self._load_training_log()
                last_channels = log[-1].get("val", {}).get("per_channel_mse", [])
                labels = self._channel_labels(len(last_channels))
                out = self._visual_path("per_channel_mse_heatmap.png")
                self.img_util.plot_per_channel_mse_from_log(log, channel_labels=labels, save_path=out)
                self.img_util.show_image_window(out, title="Per-Channel MSE Heatmap")
                return

            if vis_type == "Architecture Score":
                sg.popup(
                    "Architecture Score needs architecture-search results.\n\n"
                    "This trained-model view currently has training_log.json, but no search result list to plot.",
                    title="Architecture Score",
                )
                return

            sg.popup_error(f"Unknown visualisation type: {vis_type}")

        except Exception as e:
            print(f"[ModelViewer Visualisation Error] {vis_type}: {e}")
            sg.popup_error(f"Could not create {vis_type} visualisation:\n\n{e}")

    # ------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------
    def start_model_visual_task(self, window, visual_type):
        if not self._require_model():
            return

        window.write_event_value(
            "-TASK_MODEL_VISUALISATION-",
            {"model_name": self.cfg.model_name, "visual_type": visual_type},
        )

    def start_training(self, window):
        if not self._require_model():
            return
        print(f"[ModelViewer] Starting training for {self.cfg.model_name}")
        window.write_event_value("-TASK_TRAIN_MODEL-", self.cfg.model_name)

    def start_predictive_test(self, window):
        if not self._require_model():
            return

        if not self.selected_predictive_dataset:
            sg.popup_error("Choose a predictive/ground-truth dataset before running model testing.")
            return

        print(f"[ModelViewer] Running predictive test for {self.cfg.model_name} on {self.selected_predictive_dataset}")
        window.write_event_value(
            "-TASK_RUN_PREDICTIVE_TEST-",
            {"model_name": self.cfg.model_name, "dataset_name": self.selected_predictive_dataset},
        )

    def start_threshold_calibration(self, window):
        if not self._require_model():
            return

        if not self.selected_predictive_dataset:
            sg.popup_error("Choose a predictive/ground-truth dataset before calibrating the threshold.")
            return

        print(f"[ModelViewer] Calibrating threshold for {self.cfg.model_name} on {self.selected_predictive_dataset}")
        window.write_event_value(
            "-TASK_CALIBRATE_THRESHOLD-",
            {"model_name": self.cfg.model_name, "dataset_name": self.selected_predictive_dataset},
        )

    def start_evaluation(self, window):
        if not self._require_model():
            return

        if not self.selected_evaluation_dataset:
            sg.popup_error("Choose an evaluation/discovery dataset before finding anomalies.")
            return

        print(f"[ModelViewer] Running evaluation/discovery for {self.cfg.model_name} on {self.selected_evaluation_dataset}")
        window.write_event_value(
            "-TASK_RUN_EVALUATION-",
            {"model_name": self.cfg.model_name, "dataset_name": self.selected_evaluation_dataset},
        )

    def update_profile_status(self, compatibility):
        if not self.txt_profile_status:
            return

        if not compatibility:
            self.txt_profile_status.update("Profile Check: -")
            return

        if compatibility.get("compatible"):
            self.txt_profile_status.update("Profile Check: compatible")
        else:
            reason = compatibility.get("reason", "profiles do not match")
            self.txt_profile_status.update(f"Profile Check: NOT compatible ({reason})")

    def apply_stage(self, stage):
        self.views["info"].update(visible=True)
        self.views["visuals"].update(visible=False)
        self.views["dataset"].update(visible=False)
        self.views["training"].update(visible=False)
        self.views["prediction"].update(visible=False)

        if stage == "created":
            self.views["dataset"].update(visible=True)
        elif stage == "training":
            self.views["training"].update(visible=True)
        elif stage == "trained":
            self.views["visuals"].update(visible=True)
            self.views["prediction"].update(visible=True)
        elif stage == "searching":
            self.views["visuals"].update(visible=True)
        elif stage == "completed":
            self.views["visuals"].update(visible=True)
            self.views["training"].update(visible=True)
            self.views["prediction"].update(visible=True)
        else:
            self.views["dataset"].update(visible=True)

        self.current_stage = stage

    def on_worker_message(self, task_id, msg_type, data):
        match msg_type:
            case "status":
                print(f"[MODEL STATUS] {data}")
            case "progress":
                print(f"[MODEL PROGRESS] {data}")
            case "result":
                print(f"[MODEL RESULT] {data}")
            case "error":
                print(f"[MODEL ERROR] {data}")
            case "finished":
                print(f"[MODEL TASK FINISHED] {task_id}")
