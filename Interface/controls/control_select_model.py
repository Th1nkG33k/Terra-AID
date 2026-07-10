import PySimpleGUI as sg

from Interface.controls.control_select_base import ControlSelectBase
from Interface.theme import RPanel, RText, COLORS, RButton
from Core.Managers.model_manager import ModelManager


class ControlSelectModel(ControlSelectBase):

    key = "-CONTROL_SELECT_MODEL-"

    def __init__(self, model_manager=None):

        super().__init__()
        self.title = "Select Model"
        self.button_load_key = "-CTL_SEL_MODEL_LOAD-"
        self.button_cancel_key = "-CTL_SEL_MODEL_CANCEL-"

        # ------------------------------------------------------------------
        # ModelManager is normally supplied by AppContext.
        # Fall back to creating one so older callers still work during refactor.
        # ------------------------------------------------------------------
        
        self.mm = model_manager or ModelManager()

        # Build list of models
        self.model_list = self._build_model_list()


    # ------------------------------------------------------------
    # Build list of models from configs
    # ------------------------------------------------------------
    def _build_model_list(self):
        items = []

        for name in self.mm.list_models():
            cfg = self.mm.get(name)

            epochs = cfg.training.epochs
            device = cfg.device
            ds_name = getattr(cfg, "training_dataset", None)
            channels = "?"
            if ds_name and hasattr(self.mm, "config_manager"):
                ds = self.mm.config_manager.get_dataset(ds_name)
                channels = getattr(ds, "num_input_channels", None) or "?"
            profile = f"derived_{channels}ch" if channels != "?" else "no dataset"

            label = f"{epochs} epochs | {device} | {profile} | {channels} ch"
            print(f"[ControlSelectModel] {name}: {label}")

            items.append((name, label))

        return items


    # ------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------
    def build(self, window):
        return super().build(window, self.model_list)


    # ------------------------------------------------------------
    # Render a model item
    # ------------------------------------------------------------
    def build_item(self, data, idx):

        name, label = data

        dets_panel = RPanel(
            key=f"{self.key}_ITEM_{idx}_DETS_PANEL",
            layout=[
                [RText(name,  key=f"{self.key}_ITEM_{idx}_NAME")],
                [RText(label, key=f"{self.key}_ITEM_{idx}_LABEL")],
            ],
            w=0.68,
        )

        btn = RButton("Select",
                      key=f"{self.key}_ITEM_{idx}",
                      w=0.90,
                      pad=(5, 5)
        )

        btn_panel = RPanel(key=f"{self.key}_ITEM_{idx}_BTN_PANEL",
                           layout=[[btn]],
                           w=0.22,
        )

        return RPanel(key=f"{self.key}_ITEM_{idx}_PANEL",
                      layout=[[dets_panel, btn_panel]],
                      w=0.95,
        )


    # ------------------------------------------------------------
    # Return selected model name
    # ------------------------------------------------------------
    def get_selected_model(self):

        if self.selected_index is None:
            return None

        name, _ = self.model_list[self.selected_index]
        return name


    # ------------------------------------------------------------
    # Popup window
    # ------------------------------------------------------------
    def show(self, parent_window):

        control_layout = self.build(parent_window)

        win = sg.Window(self.title,
                        [[control_layout]],
                        modal=True,
                        keep_on_top=True,
                        finalize=True,
                        background_color=COLORS["bg_dark"],
                        size=(560, 440),
        )

        selected_value = None

        while True:

            event, values = win.read()

            if event in (sg.WIN_CLOSED, self.button_cancel_key):
                break

            # SELECT button
            if event.startswith(f"{self.key}_ITEM_") and event.count("_") == 4:

                idx = int(event.split("_")[-1])
                self.selected_index = idx

                # highlight selected
                for i in range(len(self.model_list)):
                    
                    win[f"{self.key}_ITEM_{i}"].update(
                        button_color=(COLORS["accent_teal"], COLORS["bg_panel"])
                    )

                win[f"{self.key}_ITEM_{idx}"].update(
                    button_color=(COLORS["accent_teal"], COLORS["bg_panel"])
                )

                selected_value = self.get_selected_model()
                break

        win.close()
        return selected_value
