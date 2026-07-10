

import PySimpleGUI as sg
import inspect

from Interface.controls.control_select_base import ControlSelectBase
from ..theme import RPanel, RText, COLORS, FONTS, RButton, RDSPanel
from Core.Managers.dataset_manager import DatasetManager


class ControlSelectDataset(ControlSelectBase):

    key = "-CONTROL_SELECT_DATASET-"

    def __init__(self, dataset_manager=None, mode="all"):

        super().__init__()
        self.mode = mode
        self.title = self._title_for_mode(mode)
        self.button_load_key = "-CTL_SEL_DS_LOAD-"
        self.button_cancel_key = "-CTL_SEL_DS_CANCEL-"

        # DatasetManager is normally supplied by AppContext.
        # Fall back to creating one so older callers still work during refactor.
        self.dm = dataset_manager or DatasetManager()

        self.dataset_list = self._build_dataset_list()


    # ------------------------------------------------------------
    # Return the popup title for the current selector role.
    # ------------------------------------------------------------
    def _title_for_mode(self, mode):

        titles = {"training": "Select Training Dataset",
                  "predictive": "Select Predictive / Ground-Truth Dataset",
                  "prediction": "Select Predictive / Ground-Truth Dataset",
                  "evaluation": "Select Evaluation / Discovery Dataset",
                  "validation": "Select Predictive / Ground-Truth Dataset",
                  "all": "Select Dataset",
        }

        return titles.get((mode or "all").lower(), "Select Dataset")


    # ------------------------------------------------------------
    # Build list of datasets from configs
    # ------------------------------------------------------------
    def _build_dataset_list(self):
        items = []

        for opt in self.dm.list_dataset_options(role=self.mode):
            name = opt["key"]
            tile_count = opt.get("tile_count") or "?"
            stage = opt.get("stage") or "unknown"
            role = opt.get("role") or "mixed"
            structure = opt.get("structure") or "aoi_grid"
            channels = opt.get("num_input_channels") or "?"

            label = f"{tile_count} tiles | {stage} | {role} | {structure} | {channels} model ch"
            items.append((name, label))
        
        return items


    # ------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------
    def build(self, window):

        return super().build(window, self.dataset_list)


    # ------------------------------------------------------------
    # Render a dataset item
    # ------------------------------------------------------------
    def build_item(self, data, idx):

        name, label = data

        if "multi" in name:
            # Left side: name + label
            dets_panel = RDSPanel(key=f"{self.key}_ITEM_{idx}_DETS_PANEL",
                                  layout=[
                                            [RText(name,  key=f"{self.key}_ITEM_{idx}_NAME")],
                                            [RText(label, key=f"{self.key}_ITEM_{idx}_LABEL")],
                                  ],
                                  w=0.68,
            )

            # Clickable button
            btn = RButton("Select",
                          key=f"{self.key}_ITEM_{idx}",
                          w=0.90,
                          pad=(5,5),
            )

            btn_panel = RDSPanel(key=f"{self.key}_ITEM_{idx}_BTN_PANEL",
                               layout=[[btn]],          
                               w=0.22,
            )


            # Parent panel returned to base class
            item_panel = RDSPanel(key=f"{self.key}_ITEM_{idx}_PANEL",
                                  layout=[[dets_panel, btn_panel]],   # ✔️ FIXED
                                  w=0.95,
            )
        else:
            # Left side: name + label
            dets_panel = RPanel(key=f"{self.key}_ITEM_{idx}_DETS_PANEL",
                                layout=[
                                          [RText(name,  key=f"{self.key}_ITEM_{idx}_NAME")],
                                          [RText(label, key=f"{self.key}_ITEM_{idx}_LABEL")],
                                ],
                                w=0.68,
            )

            # Clickable button
            btn = RButton("Select",
                        key=f"{self.key}_ITEM_{idx}",
                        w=0.90,
                        pad=(5,5),
            )

            btn_panel = RPanel(key=f"{self.key}_ITEM_{idx}_BTN_PANEL",
                            layout=[[btn]],          
                            w=0.22,
            )

            # Parent panel returned to base class
            item_panel = RPanel(key=f"{self.key}_ITEM_{idx}_PANEL",
                                layout=[[dets_panel, btn_panel]],   # ✔️ FIXED
                                w=0.95,
            )


        return item_panel


    # ------------------------------------------------------------
    # What to return when Load Selected is pressed
    # Returns the dataset name selected by the user.
    # ------------------------------------------------------------
    def get_selected_dataset(self):

        if self.selected_index is None:
            return None

        name, _ = self.dataset_list[self.selected_index]
        
        return name


    # -------------------------------------------------------
    #    Opens the control as a modal popup window.
    #    Blocks until user selects an item or cancels.
    #    Returns the selected dataset name (or None).
    # -------------------------------------------------------
    def show(self, parent_window):

        # self.build() already returns a Column or Frame layout
        control_layout = self.build(parent_window)

        win = sg.Window(self.title,
                        [[control_layout]],   # <-- THIS is the correct wrapping
                        modal=True,
                        keep_on_top=True,
                        finalize=True,
                        background_color=COLORS["bg_dark"],
                        size=(620, 440),
        )

        selected_value = None
        
        while True:
            event, values = win.read()

            if event in (sg.WIN_CLOSED, self.button_cancel_key):
                break

            # --- SELECT button clicked ---
            if event.startswith(f"{self.key}_ITEM_") and event.count("_") == 4:
                idx = int(event.split("_")[-1])
                self.selected_index = idx

                # highlight
                for i in range(len(self.dataset_list)):
                    win[f"{self.key}_ITEM_{i}"].update(
                        button_color=(COLORS["accent_teal"], COLORS["bg_panel"])
                    )

                win[f"{self.key}_ITEM_{idx}"].update(
                    button_color=(COLORS["accent_teal"], COLORS["bg_panel"])
                )

                # CLOSE + RETURN
                selected_value = self.get_selected_dataset()
                print(selected_value)
                break

        win.close()
        return selected_value
