
import re

import PySimpleGUI as sg
from Interface.theme import (RPanel, RButton, RText, RMultiline, COLORS)


# ============================================================
# DATASET TASK
#
#   This is NOT a PageManager page.
#   It is a standalone task window used for long‑running operations.  
# ============================================================
class PageCreateDatasetTask:

    def __init__(self):

        self.progress_bar = None
        self.status_text = None
        self.log_box = None
        self.progress_text = None
        self.window = None
        self.had_error = False
        self.is_cancelling = False

    # -------------------------
    # BUILD TASK WINDOW
    #   Creates and returns a NEW WINDOW dedicated to the task.
    #   This window is NOT part of the main layout.
    # -------------------------
    def open(self):
        # -------------------------
        # STATUS TEXT
        # -------------------------
        self.status_text = RText("Waiting for task…", key="-CDT_STATUS-", w=0.40)

        status_panel = RPanel(key="-CDT_STATUS_PANEL-",
                              layout=[[self.status_text]],
                              w=0.40,
        )

        # -------------------------
        # PROGRESS BAR
        # -------------------------
        self.progress_bar = sg.ProgressBar(max_value=100,
                                           orientation="h",
                                           size=(40, 20),
                                           key="-CDT_PROGRESS-",
                                           bar_color=("cyan", COLORS["bg_panel"]),
                                           visible=True,
                                           relief=sg.RELIEF_SUNKEN,
        )

        self.progress_text = RText("0%", key="-CDT_PROGRESS_TEXT-", w=0.05)

        progress_panel = RPanel(key="-CDT_PROGRESS_PANEL-",
                                layout=[
                                        [
                                            self.progress_bar,
                                            self.progress_text,
                                        ]
                                ],
                                w=0.50,
        )

        # -------------------------
        # LOG OUTPUT
        # -------------------------
        self.log_box = RMultiline("-CDT_LOGS-", w=0.60, h=0.30)

        logs_panel = RPanel(key="-CDT_LOGS_PANEL-",
                            layout=[
                                    [RText("Task Log")],
                                    [self.log_box],
                            ],
                            w=0.60,
        )

        # -------------------------
        # ACTION BUTTONS
        # -------------------------
        actions_panel = RPanel(key="-CDT_ACTIONS_PANEL-",
                               layout=[
                                        [
                                            sg.Push(),
                                            RButton("Cancel", key="-CDT_CANCEL-", w=0.12),
                                            sg.Push(),
                                        ]
                               ],
                               w=1.00,
        )

        # -------------------------
        # WINDOW LAYOUT
        # -------------------------
        layout = [
 #                   [title_panel],
                    [status_panel],
                    [progress_panel],
                    [logs_panel],
                    [actions_panel],
        ]

        # -------------------------
        # CREATE NEW WINDOW
        # -------------------------
        self.window = sg.Window("Dataset Task",
                                [[sg.Column(layout, background_color=COLORS["bg_dark"])]],
                                modal=True,
                                finalize=True,
                                resizable=True,
                                background_color=COLORS["bg_dark"],
        )

        return self.window

    # ------------------------------------------------------------
    # Extract a percentage from worker progress payloads.
    # Supports integers, floats, dictionaries, and strings such as
    # "Processing 42% — tile 0".
    # ------------------------------------------------------------
    def _extract_percent(self, data):

        if isinstance(data, dict):
            data = data.get("percent", data.get("pct", data.get("progress")))

        if isinstance(data, (int, float)):
            return max(0, min(100, int(data)))

        if isinstance(data, str):
            match = re.search(r"(\d{1,3})\s*%", data)
            if match:
                return max(0, min(100, int(match.group(1))))

            try:
                return max(0, min(100, int(float(data))))
            except ValueError:
                return None

        return None


    # ------------------------------------------------------------
    # Update progress bar and percentage label together.
    # ------------------------------------------------------------
    def _update_progress(self, pct):

        self.progress_bar.update_bar(pct)

        if self.progress_text is not None:
            self.progress_text.update(f"{pct}%")




    # ------------------------------------------------------------
    # Mark the task as cancelling while the background worker exits.
    # ------------------------------------------------------------
    def mark_cancelling(self):

        self.had_error = True
        self.is_cancelling = True

        if self.status_text is not None:
            self.status_text.update("Cancelling task…")

        if self.log_box is not None:
            self.log_box.update("Cancellation requested. Waiting for the task to stop safely...\n", append=True)

        if self.window is not None and "-CDT_CANCEL-" in self.window.AllKeysDict:
            self.window["-CDT_CANCEL-"].update(disabled=True)


    # ------------------------------------------------------------
    # WORKER MESSAGE HANDLER
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):

        match msg_type:

            case "status":
                self.status_text.update(data)
                self.log_box.update(f"{data}\n", append=True)

            case "progress":
                pct = self._extract_percent(data)

                if pct is not None:
                    self._update_progress(pct)
                    self.status_text.update(f"{pct}%")

                self.log_box.update(f"{data}\n", append=True)

            case "result":
                message = data.get("message", "Task completed successfully.") if isinstance(data, dict) else "Task completed successfully."
                self.status_text.update(message)
                self._update_progress(100)
                self.log_box.update(f"{message}\n", append=True)

            case "error":
                self.had_error = True
                self.status_text.update("Error occurred")
                self.log_box.update(f"ERROR:\n{data}\n", append=True)
                sg.popup_error("Dataset Task Error", data)

            case "finished":
                if not self.had_error:
                    self.status_text.update("Task finished")
                    self._update_progress(100)

                self.log_box.update("Task finished.\n", append=True)

                if (not self.had_error or self.is_cancelling) and self.window is not None:
                    self.window.close()
                    self.window = None

