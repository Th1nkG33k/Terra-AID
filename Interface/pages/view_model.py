import json
from pathlib import Path

import PySimpleGUI as sg

from Interface.theme import COLORS, FONTS, RText, RHText, RButton
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

        # Internal names are kept backwards-compatible with the existing
        # task/workflow code. The UI now presents these as:
        #   predictive dataset -> Evaluation Dataset (labelled / ground truth)
        #   evaluation dataset -> Prediction Dataset (unlabelled anomaly search)
        self.selected_predictive_dataset = None
        self.selected_evaluation_dataset = None

        self.info_fields = {}
        self.txt_evaluation_dataset = None
        self.txt_prediction_dataset = None
        self.txt_profile_status = None

        self.training_select_dataset_col = None
        self.training_train_col = None
        self.training_visuals_col = None
        self.evaluation_actions_col = None
        self.prediction_actions_col = None

    # ------------------------------------------------------------
    # Small layout helpers
    # ------------------------------------------------------------
    def _info_value(self, key, width=38):
        text = sg.Text(
            "-",
            key=key,
            size=(width, 1),
            font=FONTS["body"],
            text_color=COLORS["text_primary"],
            background_color=COLORS["bg_dark"],
        )
        self.info_fields[key] = text
        return text

    def _info_row(self, label, key, width=38):
        return [
            sg.Text(
                label,
                size=(18, 1),
                font=FONTS["body"],
                text_color=COLORS["text_secondary"],
                background_color=COLORS["bg_dark"],
                justification="right",
            ),
            self._info_value(key, width=width),
        ]

    def _asset_path(self, filename):
        if not filename:
            return None
        return Path(__file__).resolve().parents[2] / "assets" / filename

    def _visual_button(self, text, key, image_filename=None):
        """
        Clickable visualisation tile. If the named image exists in /assets it is
        used as the button image; otherwise a neat text button is used. This lets
        image assets be added later without another code refactor.
        """
        image_path = self._asset_path(image_filename)
        common = {
            "key": key,
            "font": FONTS["body"],
            "button_color": (COLORS["text_primary"], COLORS["bg_panel"]),
            "mouseover_colors": (COLORS["text_primary"], COLORS["accent_teal"]),
            "border_width": 1,
            "pad": (8, 8),
        }

        if image_path and image_path.exists():
            return sg.Button(
                "",
                image_filename=str(image_path),
                image_size=(165, 95),
                **common,
            )

        return sg.Button(text, size=(18, 4), **common)

    # ------------------------------------------------------------
    # Build static layout
    # ------------------------------------------------------------
    def build_views(self):
        # -----------------------------
        # Professional two-column header
        # -----------------------------
        left_info = sg.Column(
            [
                self._info_row("Name", f"{self.key}_INFO_NAME"),
                self._info_row("Stage", f"{self.key}_INFO_STAGE"),
                self._info_row("Architecture", f"{self.key}_INFO_ARCH"),
                self._info_row("Input Channels", f"{self.key}_INFO_CHANNELS"),
                self._info_row("Training Dataset", f"{self.key}_INFO_TRAIN_DS"),
                self._info_row("Model Inputs", f"{self.key}_INFO_INPUTS", width=52),
            ],
            background_color=COLORS["bg_dark"],
            pad=((0, 25), (0, 0)),
            vertical_alignment="top",
        )

        right_info = sg.Column(
            [
                self._info_row("Optimizer", f"{self.key}_INFO_OPT"),
                self._info_row("Learning Rate", f"{self.key}_INFO_LR"),
                self._info_row("Weight Decay", f"{self.key}_INFO_WEIGHT_DECAY"),
                self._info_row("Training", f"{self.key}_INFO_TRAINING", width=52),
                self._info_row("Scheduler", f"{self.key}_INFO_SCHEDULER"),
                self._info_row("Architecture Config", f"{self.key}_INFO_ARCH_CFG", width=52),
            ],
            background_color=COLORS["bg_dark"],
            pad=((25, 0), (0, 0)),
            vertical_alignment="top",
        )

        info_layout = [
            [RHText("Information")],
            [left_info, sg.VSeparator(color=COLORS["line_bright"]), right_info],
        ]

        # -----------------------------
        # Training tab
        # -----------------------------
        self.training_select_dataset_col = sg.Column(
                [
                    [RText("Training Dataset")],
                    [RText("Choose the processed dataset used to train this model.", color=COLORS["text_secondary"])],
                    [RButton("Select Training Dataset", key=f"{self.key}_TRAIN_DATASET", w=0.22)],
                ],
                key=f"{self.key}_TRAINING_SELECT_DATASET_PANEL",
                background_color=COLORS["bg_dark"],
                visible=False,
                pad=(0, 10),
            )

        self.training_train_col = sg.Column(
                [
                    [RText("Training")],
                    [RText("Run the selected model architecture against the assigned training dataset.", color=COLORS["text_secondary"])],
                    [RButton("Train Model", key=f"{self.key}_TRAIN", w=0.18)],
                ],
                key=f"{self.key}_TRAINING_RUN_PANEL",
                background_color=COLORS["bg_dark"],
                visible=False,
                pad=(0, 10),
            )

        self.training_visuals_col = sg.Column(
                [
                    [RText("Visualisations")],
                    [
                        self._visual_button("Results", f"{self.key}_RESULTS", "model_results.png"),
                        self._visual_button("Loss Curve", f"{self.key}_LOSS_CURVE", "model_loss_curve.png"),
                        self._visual_button("Loss Metrics", f"{self.key}_LOSS_METRICS", "model_loss_metrics.png"),
                        self._visual_button("Channel Heatmap", f"{self.key}_HEATMAP", "model_channel_heatmap.png"),
                    ],
                    [
                        self._visual_button("Architecture Score", f"{self.key}_ARCH_SCORE", "model_arch_score.png"),
                        self._visual_button("Anomaly Map", f"{self.key}_ANOMALY_MAP", "model_anomaly_map.png"),
                        self._visual_button("Clustering", f"{self.key}_CLUSTER", "model_clustering.png"),
                    ],
                ],
                key=f"{self.key}_TRAINING_VISUALS_PANEL",
                background_color=COLORS["bg_dark"],
                visible=False,
                pad=(0, 10),
            )

        training_tab_layout = [
            [self.training_select_dataset_col],
            [self.training_train_col],
            [self.training_visuals_col],
        ]

        # -----------------------------
        # Evaluation tab: labelled ground-truth model evaluation
        # -----------------------------
        self.txt_evaluation_dataset = RText("Evaluation Dataset: -")
        self.evaluation_actions_col = sg.Column(
                [
                    [RText("Evaluation")],
                    [RText("Use a labelled ground-truth dataset to evaluate model performance and calibrate thresholds.", color=COLORS["text_secondary"])],
                    [self.txt_evaluation_dataset],
                    [
                        RButton("Select Evaluation Dataset", key=f"{self.key}_PREDICTIVE_DATASET", w=0.24),
                        RButton("Evaluate + Metrics", key=f"{self.key}_PREDICTIVE_TEST", w=0.20),
                        RButton("Calibrate Threshold", key=f"{self.key}_CALIBRATE_THRESHOLD", w=0.20),
                    ],
                ],
                key=f"{self.key}_EVALUATION_ACTIONS_PANEL",
                background_color=COLORS["bg_dark"],
                visible=False,
                pad=(0, 10),
            )

        evaluation_tab_layout = [[self.evaluation_actions_col]]

        # -----------------------------
        # Prediction tab: anomaly discovery on user datasets
        # -----------------------------
        self.txt_prediction_dataset = RText("Prediction Dataset: -")
        self.txt_profile_status = RText("Profile Check: -", color=COLORS["text_secondary"])
        self.prediction_actions_col = sg.Column(
                [
                    [RText("Prediction")],
                    [RText("Run the trained model against an unlabelled dataset to find potential anomalies.", color=COLORS["text_secondary"])],
                    [self.txt_prediction_dataset],
                    [
                        RButton("Select Prediction Dataset", key=f"{self.key}_EVALUATION_DATASET", w=0.24),
                        RButton("Find Anomalies", key=f"{self.key}_EVALUATE", w=0.18),
                    ],
                    [self.txt_profile_status],
                ],
                key=f"{self.key}_PREDICTION_ACTIONS_PANEL",
                background_color=COLORS["bg_dark"],
                visible=False,
                pad=(0, 10),
            )

        prediction_tab_layout = [[self.prediction_actions_col]]

        tabs = sg.TabGroup(
            [[
                sg.Tab("Training", training_tab_layout, key=f"{self.key}_TAB_TRAINING", background_color=COLORS["bg_dark"]),
                sg.Tab("Evaluation", evaluation_tab_layout, key=f"{self.key}_TAB_EVALUATION", background_color=COLORS["bg_dark"]),
                sg.Tab("Prediction", prediction_tab_layout, key=f"{self.key}_TAB_PREDICTION", background_color=COLORS["bg_dark"]),
            ]],
            key=f"{self.key}_TABGROUP",
            enable_events=True,
            background_color=COLORS["bg_dark"],
            tab_background_color=COLORS["bg_panel"],
            selected_background_color=COLORS["accent_teal"],
            selected_title_color=COLORS["text_primary"],
            title_color=COLORS["text_primary"],
            border_width=0,
            expand_x=True,
            expand_y=True,
            pad=((0, 0), (18, 0)),
        )

        main_layout = [
            *info_layout,
            [tabs],
        ]

        main_view = sg.Column(
            main_layout,
            key=f"{self.key}_MAIN",
            visible=False,
            background_color=COLORS["bg_dark"],
            expand_x=True,
            expand_y=True,
            pad=(0, 0),
        )
        self.add_view("main", main_view)

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

    def _architecture_config_text(self, cfg):
        arc = getattr(cfg, "architecture", None)
        if arc is None:
            return "-"

        arc_type = str(getattr(arc, "type", "mae"))
        if arc_type == "mae":
            parts = [
                f"Enc Depth {self._safe(getattr(arc, 'encoder_depth', None))}",
                f"Dec Depth {self._safe(getattr(arc, 'decoder_depth', None))}",
                f"Embed {self._safe(getattr(arc, 'embed_dim', None))}",
                f"Dec Dim {self._safe(getattr(arc, 'decoder_dim', None))}",
                f"Mask {self._safe(getattr(arc, 'mask_ratio', None))}",
                f"Base {self._safe(getattr(arc, 'base_channels', None))}",
            ]
        elif arc_type == "cnn_autoencoder":
            parts = [
                f"Depth {self._safe(getattr(arc, 'encoder_depth', None))}",
                f"Base {self._safe(getattr(arc, 'base_channels', None))}",
                f"Latent {self._safe(getattr(arc, 'latent_channels', None))}",
            ]
        elif arc_type == "resnet_autoencoder":
            parts = [
                f"Backbone {self._safe(getattr(arc, 'backbone', None))}",
                f"Pretrained {self._safe(getattr(arc, 'pretrained', None))}",
                f"Freeze Epochs {self._safe(getattr(arc, 'freeze_encoder_epochs', None))}",
            ]
        else:
            parts = []

        return "  |  ".join(parts) if parts else "-"

    def _update_field(self, key, value):
        field = self.info_fields.get(key)
        if field:
            field.update(str(self._safe(value)))

    # ------------------------------------------------------------
    # Load model dynamically (called from mainW)
    # ------------------------------------------------------------
    def load_model(self, cfg, window):
        previous_model_name = getattr(self.cfg, "model_name", None)
        new_model_name = getattr(cfg, "model_name", None)
        if previous_model_name != new_model_name:
            self.selected_predictive_dataset = None
            self.selected_evaluation_dataset = None

        self.model_cfg = cfg
        self.cfg = cfg
        self.window = window

        arc = getattr(cfg, "architecture", None)
        opt = getattr(cfg, "optimizer", None)
        tr = getattr(cfg, "training", None)
        sch = getattr(cfg, "scheduler", None)

        derived_channels = self._training_dataset_channels(cfg)
        channel_count = len(derived_channels) if derived_channels else getattr(arc, "num_channels", None)

        self._update_field(f"{self.key}_INFO_NAME", getattr(cfg, "model_name", None))
        self._update_field(f"{self.key}_INFO_STAGE", getattr(cfg, "stage", None))
        self._update_field(f"{self.key}_INFO_ARCH", getattr(arc, "type", None))
        self._update_field(f"{self.key}_INFO_CHANNELS", f"{self._safe(channel_count)}")
        self._update_field(f"{self.key}_INFO_TRAIN_DS", getattr(cfg, "training_dataset", None))
        self._update_field(f"{self.key}_INFO_INPUTS", f"{self._safe(channel_count)} channels  |  {self._channels_text(derived_channels)}" if derived_channels else "choose a training dataset")
        self._update_field(f"{self.key}_INFO_OPT", getattr(opt, "type", None))
        self._update_field(f"{self.key}_INFO_LR", getattr(opt, "lr", None))
        self._update_field(f"{self.key}_INFO_WEIGHT_DECAY", getattr(opt, "weight_decay", None))
        self._update_field(
            f"{self.key}_INFO_TRAINING",
            f"Batch {self._safe(getattr(tr, 'batch_size', None))}  |  Workers {self._safe(getattr(tr, 'num_workers', None))}  |  "
            f"Device {self._safe(getattr(cfg, 'device', None))}  |  Epochs {self._safe(getattr(tr, 'epochs', None))}  |  "
            f"Patience {self._safe(getattr(tr, 'early_stopping_patience', None))}",
        )
        self._update_field(
            f"{self.key}_INFO_SCHEDULER",
            f"{self._safe(getattr(sch, 'type', None))}  |  Warmup {self._safe(getattr(sch, 'warmup_epochs', None))}",
        )
        self._update_field(f"{self.key}_INFO_ARCH_CFG", self._architecture_config_text(cfg))

        self.txt_evaluation_dataset.update(f"Evaluation Dataset: {self._safe(self.selected_predictive_dataset)}")
        self.txt_prediction_dataset.update(f"Prediction Dataset: {self._safe(self.selected_evaluation_dataset)}")
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
        elif event == f"{self.key}_TABGROUP":
            handled = True
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

        # Internal role/mode is still predictive for backwards compatibility,
        # but the UI now treats this as labelled model evaluation data.
        selector = ControlSelectDataset(self.dataset_manager, mode="predictive")
        selected_dataset = selector.show(window)

        if selected_dataset:
            self.selected_predictive_dataset = selected_dataset
            self.txt_evaluation_dataset.update(f"Evaluation Dataset: {selected_dataset}")
            compatibility = self.model_manager.check_dataset_compatibility(self.cfg.model_name, selected_dataset) if self.model_manager else None
            self.update_profile_status(compatibility)

    def select_evaluation_dataset(self, window):
        if not self._require_model():
            return

        # Internal mode is still evaluation for backwards compatibility,
        # but the UI now presents it as prediction/anomaly discovery.
        selector = ControlSelectDataset(self.dataset_manager, mode="evaluation")
        selected_dataset = selector.show(window)

        if selected_dataset:
            self.selected_evaluation_dataset = selected_dataset
            self.txt_prediction_dataset.update(f"Prediction Dataset: {selected_dataset}")
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
            sg.popup_error("Choose an evaluation/ground-truth dataset before running model evaluation.")
            return

        print(f"[ModelViewer] Running model evaluation for {self.cfg.model_name} on {self.selected_predictive_dataset}")
        window.write_event_value(
            "-TASK_RUN_PREDICTIVE_TEST-",
            {"model_name": self.cfg.model_name, "dataset_name": self.selected_predictive_dataset},
        )

    def start_threshold_calibration(self, window):
        if not self._require_model():
            return

        if not self.selected_predictive_dataset:
            sg.popup_error("Choose an evaluation/ground-truth dataset before calibrating the threshold.")
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
            sg.popup_error("Choose a prediction/discovery dataset before finding anomalies.")
            return

        print(f"[ModelViewer] Running prediction/discovery for {self.cfg.model_name} on {self.selected_evaluation_dataset}")
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
        # One stable model page with a tab control. The stage controls which
        # actions are visible inside each tab rather than swapping entire pages.
        self.views["main"].update(visible=True)

        # Reset all stage-specific sections.
        self.training_select_dataset_col.update(visible=False)
        self.training_train_col.update(visible=False)
        self.training_visuals_col.update(visible=False)
        self.evaluation_actions_col.update(visible=False)
        self.prediction_actions_col.update(visible=False)

        if stage == "created":
            self.training_select_dataset_col.update(visible=True)

        elif stage == "training":
            self.training_select_dataset_col.update(visible=True)
            self.training_train_col.update(visible=True)

        elif stage in ("trained", "completed"):
            self.training_visuals_col.update(visible=True)
            self.evaluation_actions_col.update(visible=True)
            self.prediction_actions_col.update(visible=True)

        elif stage == "searching":
            self.training_visuals_col.update(visible=True)
            self.prediction_actions_col.update(visible=True)

        else:
            self.training_select_dataset_col.update(visible=True)

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
