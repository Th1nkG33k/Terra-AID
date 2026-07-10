import PySimpleGUI as sg
from Interface.theme import (RText, RButton, RPanel, RMultiline, COLORS)


# ============================================================
# HOME PAGE
# ============================================================
class PageHome:
    key = "-PAGE_HOME-"

    def __init__(self):
        self.dataset_summary = None
        self.model_summary = None
        self.logs_box = None

    def build(self, window):

        dataset_summary = RText("No dataset loaded", key="-HOME_DATASET_SUMMARY-")
        model_summary = RText("No model loaded", key="-HOME_MODEL_SUMMARY-")
        logs_box = RMultiline("-HOME_LOGS_BOX-", w=0.30, h=0.20)

        self.dataset_summary = dataset_summary
        self.model_summary = model_summary
        self.logs_box = logs_box

        dataset_panel = RPanel(
            key="-HOME_DATASET_PANEL-",
            layout=[
                [RText("Dataset Summary")],
                [dataset_summary],
            ],
            w=0.30,
        )

        model_panel = RPanel(
            key="-HOME_MODEL_PANEL-",
            layout=[
                [RText("Model Summary")],
                [model_summary],
            ],
            w=0.30,
        )

        logs_panel = RPanel(
            key="-HOME_LOGS_PANEL-",
            layout=[
                [RText("Recent Logs")],
                [logs_box],
            ],
            w=0.30,
        )

        # ---------------------------------------------------------------------
        # Use home-specific keys here.  The Dataset and Model pages keep their
        # existing selector keys, which prevents duplicate keys and keeps those
        # page links working as before.
        # ---------------------------------------------------------------------
        
        actions_row = [
            RButton("Load Dataset", "-HOME_LOAD_DATASET-", w=0.15),
            RButton("Load Model", "-HOME_LOAD_MODEL-", w=0.15),
            RButton("Logs", "-HOME_OPEN_LOGS-", w=0.15),
        ]

        layout = [
            [RText("Terra-AID Dashboard", key="-HOME_TITLE-", w=0.40)],
            [sg.HorizontalSeparator(color=COLORS["line_bright"])],
            [dataset_panel, model_panel, logs_panel],
            [sg.HorizontalSeparator(color=COLORS["line_bright"])],
            actions_row,
        ]

        return sg.Column(layout,
                         key=self.key,
                         expand_x=True,
                         expand_y=True,
                         background_color=COLORS["bg_dark"],
                         visible=False,
        )

    def _safe(self, value, default="-"):
        return default if value in (None, "") else value

    def update_dataset_summary(self, cfg):
        if not self.dataset_summary or cfg is None:
            return

        channels = getattr(cfg, "num_input_channels", getattr(cfg, "depth", "-"))
        text = (
            f"{cfg.dataset_name}\n"
            f"Stage: {self._safe(getattr(cfg, 'stage', None))}  |  Role: {self._safe(getattr(cfg, 'role', None))}\n"
            f"Tiles: {self._safe(getattr(cfg, 'tile_count', None))}  |  Model Inputs: {self._safe(channels)} ch"
        )
        self.dataset_summary.update(text)

    def update_model_summary(self, cfg):
        if not self.model_summary or cfg is None:
            return

        arc = getattr(getattr(cfg, "architecture", None), "type", "-")
        text = (
            f"{cfg.model_name}\n"
            f"Stage: {self._safe(getattr(cfg, 'stage', None))}  |  Architecture: {self._safe(arc)}\n"
            f"Training Dataset: {self._safe(getattr(cfg, 'training_dataset', None))}"
        )
        self.model_summary.update(text)

    def add_log(self, message):
        if self.logs_box:
            self.logs_box.update(f"{message}\n", append=True)

    # ------------------------------------------------------------
    # WORKER MESSAGE HANDLER
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):

        match msg_type:

            case "status":
                self.add_log(data)

            case "progress":
                self.add_log(data)

            case "result":
                if isinstance(data, dict):
                    if "dataset_summary" in data and self.dataset_summary:
                        self.dataset_summary.update(data["dataset_summary"])
                    if "model_summary" in data and self.model_summary:
                        self.model_summary.update(data["model_summary"])

                self.add_log("Task completed")

            case "error":
                self.add_log(f"ERROR: {data}")
                sg.popup_error("Task Error", data)

            case "finished":
                print(f"Task {task_id} finished")
