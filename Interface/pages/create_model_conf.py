import PySimpleGUI as sg
from Interface.theme import (RPanel, RButton, RText, COLORS, RInput, RHText)


# ============================================================
# CREATE MODEL CONFIG
#
#   Popup window used by Create Dataset Configuration to select an AOI.
#   The tkintermapview widget must be created after the PySimpleGUI window has
#   been finalised.  For that reason this class owns a popup window and a small
#   event loop rather than being embedded in the main page column.
#   
# ============================================================
class PageCreateModelConf:
    key = "-PAGE_CREATE_MODEL_CONF-"

    def __init__(self):
        pass

    def build(self, window):

        title_panel = RPanel(key="-CMC_TITLE_PANEL-", layout=[[RHText("Define Model")]])

        # ------------------------------------------------------------
        # MODEL NAME
        # ------------------------------------------------------------
        model_name_panel = RPanel(key="-CMC_MODEL_NAME_PANEL-",
                                  layout=[
                                            [
                                                RText("Model Name"),
                                                sg.Input(default_text="", key="-CMC_MODEL_NAME-", size=(50, 1)),
                                            ],
                                  ],
        )


        # ------------------------------------------------------------
        # DEVICE
        # ------------------------------------------------------------
        # Device is selected automatically at application start.
        # It is deliberately not exposed here because configs should not
        # override the system runtime device.


        # ------------------------------------------------------------
        # ARCHITECTURE SELECT
        # ------------------------------------------------------------
        arc_label_panel = RPanel(key="-CMC_ARC_LABEL-",
                                 layout=[
                                        [
                                            RText("Architecture"),
                                            sg.Combo(["Select Model", "mae", "cnn_autoencoder", "resnet_autoencoder"],
                                                     default_value="Select Model",
                                                     key="-CMC_ARC_TYPE-",
                                                     readonly=True,
                                                     size=(20,1),
                                                     enable_events=True,
                                                     change_submits=True,
                                            )
                                        ],
                                    ],
        )


        # ------------------------------------------------------------
        # SCHEDULER
        # ------------------------------------------------------------
        scheduler_panel = RPanel(key="-CMC_SCHEDULER_PANEL-",
                                 layout=[
                                            [
                                                RText("Scheduler"),
                                                sg.Combo(["None", "cosine"], 
                                                         default_value="None", 
                                                         key="-CMC_ARCH_SCHEDULER-", 
                                                         readonly=True,
                                                         size=(20,1)),
                                                RText("warmup epochs"),
                                                sg.Input(default_text="1", key="-CMC_ARCH_WARMUP-", size=(5,1)),
                                                sg.Push(),
                                            ],
                                    ],
        )

        # ------------------------------------------------------------
        # TRAINING SETTINGS
        # ------------------------------------------------------------
        training_panel_left = RPanel(
                                        key="-CMC_TRAIN_LEFT-",
                                        layout=[
                                            [
                                                RText("epochs"),
                                                sg.Push(), 
                                                sg.Input(default_text="20", key="-CMC_TRAIN_EPOCHS-", size=(5,1)),
                                                RText("batch size"), 
                                                sg.Input(default_text="1", key="-CMC_TRAIN_BATCH-", size=(5,1)),
                                                RText("num workers"),
                                                sg.Push(), 
                                                sg.Input(default_text="0", key="-CMC_TRAIN_WORKERS-", size=(5,1)),
                                            ],
                                            [ 
                                                RText("weight decay"), 
                                                sg.Input(default_text="1e-5", key="-CMC_TRAIN_WD-", size=(8,1)),
                                                RText("lr"), 
                                                sg.Push(),
                                                sg.Input(default_text="1e-4", key="-CMC_TRAIN_LR-", size=(8,1)),
                                                RText("early stop patience"), 
                                                sg.Input(default_text="5", key="-CMC_TRAIN_PATIENCE-", size=(5,1)),
                                           ]
                                        ]
        )

        training_label_panel = RPanel(key="-CMC_TRAIN_LABEL-", layout=[[RText("Training"), sg.Push()]]
        )

        t_pn = RPanel(key="-CMC_TR_PANEL-", layout=[[training_label_panel,training_panel_left]])
        
        # ------------------------------------------------------------
        # OPTIMIZER
        # ------------------------------------------------------------
        optimizer_panel = RPanel(key="-CMC_OPT_PANEL-",
                                 layout=[
                                        [
                                            RText("optimizer"), 
                                            sg.Combo(["AdamW"], 
                                                     default_value="AdamW", 
                                                     key="-CMC_TRAIN_OPTIMIZER-", 
                                                     readonly=True,
                                                     size=(20,1)),
                                        ],
                                ],
        )



# ============================================================
# CNN AUTOENCODER CONFIG PANEL
# ============================================================
        cnn_panel = RPanel(key="-CMC_CNN_PANEL-",
                           layout=[
                                    [
                                        RText("num channels"),
                                        sg.Push(),
                                        sg.Input(default_text="auto", key="-CMC_CNN_NUM_CHANNELS-", size=(6,1), disabled=True),
                                    ],
                                    [
                                        RText("base channels"),
                                        sg.Push(),
                                        sg.Input(default_text="32", key="-CMC_CNN_BASE_CHANNELS-", size=(6,1)),
                                    ],
                                    [
                                        RText("depth"),
                                        sg.Push(),
                                        sg.Input(default_text="4", key="-CMC_CNN_DEPTH-", size=(6,1)),
                                    ],
                                    [
                                        RText("latent channels"),
                                        sg.Push(),
                                        sg.Input(default_text="256", key="-CMC_CNN_LATENT_CHANNELS-", size=(6,1)),
                                    ],
                            ],
        )


# ============================================================
# RESNET AUTOENCODER CONFIG PANEL
# ============================================================
        resnet_panel = RPanel(key="-CMC_RESNET_PANEL-",
                              layout=[
                                        [
                                            RText("num channels"),
                                            sg.Push(),
                                            sg.Input(default_text="auto", key="-CMC_RESNET_NUM_CHANNELS-", size=(6,1), disabled=True),
                                        ],
                                        [
                                            RText("backbone"),
                                            sg.Push(),
                                            sg.Combo(["resnet50"],
                                                     default_value="resnet50",
                                                     key="-CMC_RESNET_BACKBONE-",
                                                     readonly=True,
                                                     size=(12,1)
                                            ),
                                        ],
                                        [
                                            RText("pretrained"),
                                            sg.Push(),
                                            sg.Combo(["true", "false"],
                                                     default_value="false",
                                                     key="-CMC_RESNET_PRETRAINED-",
                                                     readonly=True,
                                                     size=(8,1)
                                            ),
                                        ],
                                        [
                                            RText("freeze encoder epochs"),
                                            sg.Push(),
                                            sg.Input(default_text="0", key="-CMC_RESNET_FREEZE_EPOCHS-", size=(6,1)),
                                        ],
                            ],
        )


# ============================================================
# MAE ARCHITECTURE
# ============================================================
        arc_panel_left = RPanel(key="-CMC_ARC_LEFT-",
                                layout=[
                                    [
                                        RText("base channels"),
                                        sg.Push(), 
                                        sg.Input(default_text="32", key="-CMC_ARC_BASE_CHANNELS-", size=(5,1)), 
                                    ],
                                    [
                                        RText("encoder depth"),
                                        sg.Push(), 
                                        sg.Input(default_text="5", key="-CMC_ARC_ENCODER_DEPTH-", size=(5,1)),
                                    ],
                                    [
                                        RText("decoder depth"),
                                        sg.Push(), 
                                        sg.Input(default_text="3", key="-CMC_ARC_DECODER_DEPTH-", size=(5,1)), 
                                    ],
                                ],
        )

        arc_panel_right = RPanel(
                                    key="-CMC_ARC_RIGHT-",
                                    layout=[
                                        [
                                            RText("embed dim"),
                                            sg.Push(), 
                                            sg.Input(default_text="128", key="-CMC_ARC_EMBED_DIM-", size=(5,1)),
                                        ],
                                        [
                                            RText("decoder dim"), 
                                            sg.Push(),
                                            sg.Input(default_text="64", key="-CMC_ARC_DECODER_DIM-", size=(5,1)), 
                                        ],

                                        [
                                            RText("mask ratio"), 
                                            sg.Push(),
                                            sg.Input(default_text="0.75", key="-CMC_ARC_MASK_RATIO-", size=(5,1)), 
                                        ],
                                    ],
        )

        mae_panel = RPanel(key="-CMC_ARC_ROW_PANEL-", layout=[[arc_panel_right, arc_panel_left]])

        # ------------------------------------------------------------
        # ACTION BUTTON
        # ------------------------------------------------------------
        action_panel = RPanel(key="-CMC_ACTION_PANEL-",
                              layout=[
                                        [
                                            sg.Push(),
                                            # Create & Train → task event (NOT a page)
                                            RButton("Create", "-TASK_CREATE_MODEL-", w=0.10),
                                        ]
                                    ],
        )

        # ------------------------------------------------------------
        # combo control views
        # ------------------------------------------------------------
        mae_view = sg.Column([[mae_panel]],
                             key="-VIEW_MAE-",
                             visible=False,
                             background_color=COLORS["bg_dark"],
        )

        cnn_view = sg.Column([[cnn_panel]],
                             key="-VIEW_CNN-",
                             visible=False,
                             background_color=COLORS["bg_dark"],
        )

        resnet_view = sg.Column([[resnet_panel]],
                                key="-VIEW_RESNET-",
                                visible=False,
                                background_color=COLORS["bg_dark"],
                            )
        

        # ------------------------------------------------------------
        # PAGE LAYOUT
        # ------------------------------------------------------------
        layout = [
                    [title_panel],
                    [model_name_panel, sg.Push()],
                    [arc_label_panel],
                    [sg.pin(mae_view)],
                    [sg.pin(cnn_view)],
                    [sg.pin(resnet_view)],
                    [scheduler_panel],
                    [t_pn],
                    [optimizer_panel],
                    [action_panel],
        ]

        return sg.Column(layout,
                         key=self.key,
                         expand_x=True,
                         expand_y=True,
                         background_color=COLORS["bg_dark"],
                         visible=False,
        )

    def set_model_type(self, type, window):

        # Hide all
        window["-VIEW_MAE-"].update(visible=False)
        window["-VIEW_CNN-"].update(visible=False)
        window["-VIEW_RESNET-"].update(visible=False)

        match type:
            case "mae":
                window["-VIEW_MAE-"].update(visible=True)

            case "cnn_autoencoder":
                window["-VIEW_CNN-"].update(visible=True)

            case "resnet_autoencoder":
                window["-VIEW_RESNET-"].update(visible=True)


    # ---------------------------------------------------------
    # PAGE EVENT HANDLER
    # ---------------------------------------------------------
    def handle_event(self, event, values, window):

        if event == "-CMC_ARC_TYPE-":
            self.set_model_type(values.get("-CMC_ARC_TYPE-"), window)
            return True

        return False
