
import PySimpleGUI as sg
import Interface.theme as theme

from Core.Managers.app_context import AppContext
from Core.Tasks.dataset_tasks import (load_dataset_task, process_dataset_task, generate_statistics_task)
from Core.Tasks.model_tasks import (train_model_task, run_prediction_task, calibrate_prediction_threshold_task, model_visualisation_task)
from Core.Utils.image_utility import ImageUtility

from Interface.controls.control_select_dataset import ControlSelectDataset
from Interface.controls.control_select_model import ControlSelectModel
from Interface.pages.create_dataset_conf import PageCreateDatasetConf
from Interface.pages.create_dataset_task import PageCreateDatasetTask
from Interface.pages.create_model_conf import PageCreateModelConf
from Interface.pages.create_model_task import PageCreateModelTask
from Interface.pages.datasets import PageDatasets
from Interface.pages.home import PageHome
from Interface.pages.models import PageModels
from Interface.pages.view_dataset import DatasetViewer
from Interface.pages.view_model import ModelViewer
from Interface.theme import (apply_terra_theme, COLORS, FONTS,
                             RBannerImage, RText, update_responsive_components,)


# ------------------------------------------------------------
#  APPLICATION SETUP
# ------------------------------------------------------------
apply_terra_theme()

placeholder_layout = [
                        [
                            sg.Text("v-2.3",
                                    background_color=COLORS["bg_dark"],
                                    font=FONTS["mono"],
                                    text_color=COLORS["line_dim"],
                            )
                        ]
]

window = sg.Window("Terra-AId",
                   placeholder_layout,
                   resizable=True,
                   size=(1100, 600),
                   enable_window_config_events=True,
                   background_color=COLORS["bg_dark"],
                   finalize=True,
)

app = AppContext()
path_manager = app.paths
dataset_manager = app.datasets
model_manager = app.models
page_manager = app.pages
sidebar_manager = app.sidebar
task_manager = app.tasks


# ------------------------------------------------------------
#  PLACEHOLDER PAGES
# ------------------------------------------------------------
def page_logs(window):

    return sg.Column([[RText("Logs Page Coming Soon")]],
                     key="-PAGE_LOGS-",
                     expand_x=True,
                     expand_y=True,
                     background_color=COLORS["bg_dark"],
                     visible=False,
    )


def page_help(window):

    return sg.Column([[RText("Help Page Coming Soon")]],
                     key="-PAGE_HELP-",
                     expand_x=True,
                     expand_y=True,
                     background_color=COLORS["bg_dark"],
                     visible=False,
    )


# ------------------------------------------------------------
#  PAGE INSTANCES
# ------------------------------------------------------------
page_home = PageHome()
page_datasets = PageDatasets(dataset_manager=dataset_manager)
page_models = PageModels(model_manager=model_manager)
viewer_dataset = DatasetViewer()
viewer_model = ModelViewer(dataset_manager=dataset_manager, model_manager=model_manager)
page_create_dataset_conf = PageCreateDatasetConf()
page_create_model_conf = PageCreateModelConf()


# ------------------------------------------------------------
#  PAGE REGISTRATION
# ------------------------------------------------------------
page_manager.register("-PAGE_HOME-", page_home.build(window), handler=page_home)
page_manager.register("-PAGE_DATASETS-", page_datasets.build(window), handler=page_datasets)
page_manager.register("-PAGE_MODELS-", page_models.build(window), handler=page_models)
page_manager.register("-PAGE_LOGS-", page_logs(window))
page_manager.register("-PAGE_HELP-", page_help(window))
page_manager.register("-PAGE_CREATE_DATASET_CONF-", page_create_dataset_conf.build(window), handler=page_create_dataset_conf)
page_manager.register("-PAGE_CREATE_MODEL_CONF-", page_create_model_conf.build(window), handler=page_create_model_conf)
page_manager.register("-PAGE_VIEWER_DATASET-", viewer_dataset.build(), handler=viewer_dataset)
page_manager.register("-PAGE_VIEWER_MODEL-", viewer_model.build(), handler=viewer_model)


# ------------------------------------------------------------
#  SIDEBAR
# ------------------------------------------------------------
sidebar_manager.add("Home", "-PAGE_HOME-")
sidebar_manager.add("Datasets", "-PAGE_DATASETS-")
sidebar_manager.add("Models", "-PAGE_MODELS-")
sidebar_manager.add("Logs", "-PAGE_LOGS-")
sidebar_manager.add("Help", "-PAGE_HELP-")


# ------------------------------------------------------------
#  LAYOUT
# ------------------------------------------------------------
header_bar = [
    [
        sg.Push(),
        sg.Frame("",
                 [[RBannerImage(path_manager.banner("Terra-AID-Header_UI.png"),
                                key="-BANNER_IMAGE-",
                                w=1.00,
                 )]],
                 pad=(0, 0),
                 background_color=COLORS["bg_dark"],
                 border_width=0,
        ),
        sg.Push(),
    ]
]

sidebar_col = sg.Column(sidebar_manager.layout(),
                        key="-SIDEBAR-",
                        background_color=COLORS["bg_dark"],
                        pad=(0, 0),
                        expand_y=False,
                        expand_x=False,
                        size=(150, 400),
                        vertical_alignment="top",
)

content_row = [sg.pin(page_manager.pages[key]) for key in page_manager.pages]

content_area = sg.Column([content_row],
                         key="-CONTENT-",
                         expand_x=True,
                         expand_y=True,
                         background_color=COLORS["bg_dark"],
                         pad=(0, 0),
                         scrollable=False,
)

final_layout = [
    *header_bar,
    [sidebar_col, content_area],
]

window.extend_layout(window, final_layout)
window.refresh()

page_manager.show("-PAGE_HOME-")
sidebar_manager.set_active("-PAGE_HOME-")

theme.RESPONSIVE_READY = True
update_responsive_components(window)


# ------------------------------------------------------------
#  TASK WINDOWS
# ------------------------------------------------------------
active_task_windows = {"train_model": None,
                       "model_anomaly_map": None,
                       "model_clustering": None,
                       "predictive_test": None,
                       "evaluation_discovery": None,
                       "run_prediction": None,
                       "calibrate_threshold": None,
                       "process_dataset": None,
                       "generate_statistics": None,
}




# ------------------------------------------------------------
#  VIEW HELPERS
# ------------------------------------------------------------
def show_dataset_view(dataset_name):

    dataset_manager.reload()
    cfg = dataset_manager.get(dataset_name)

    if cfg is None:
        raise RuntimeError(f"Dataset not found after reload: {dataset_name}")

    viewer_dataset.load_dataset(cfg, window)
    page_home.update_dataset_summary(cfg)
    page_home.add_log(f"Loaded dataset: {dataset_name}")
    page_manager.show("-PAGE_VIEWER_DATASET-")
    sidebar_manager.set_active(None)
    update_responsive_components(window)
    window.refresh()

    return cfg


def show_model_view(model_name, stage=None):

    model_manager.reload()
    cfg = model_manager.get(model_name)

    if cfg is None:
        raise RuntimeError(f"Model not found after reload: {model_name}")

    viewer_model.load_model(cfg, window)
    page_home.update_model_summary(cfg)
    page_home.add_log(f"Loaded model: {model_name}")
    viewer_model.apply_stage(stage or cfg.stage)
    page_manager.show("-PAGE_VIEWER_MODEL-")
    sidebar_manager.set_active(None)
    update_responsive_components(window)
    window.refresh()

    return cfg


# ------------------------------------------------------------
#  CONTROL EVENTS
# ------------------------------------------------------------
def handle_control_event(event, values):

    # Embedded selector events on the main Datasets page.
    if page_datasets.is_filter_event(event):
        page_datasets.apply_filters(window, values)
        return True

    selected_dataset = page_datasets.selected_dataset_from_event(event)
    if selected_dataset:
        show_dataset_view(selected_dataset)
        return True

    # Embedded selector events on the main Models page.
    if page_models.is_filter_event(event):
        page_models.apply_filters(window, values)
        return True

    selected_model = page_models.selected_model_from_event(event)
    if selected_model:
        show_model_view(selected_model)
        return True

    # ---------------------------------------------------------------------
    # Home-page shortcuts still use the modal selectors. This keeps the
    # home page quick links working without reusing the embedded page keys.
    # ---------------------------------------------------------------------

    if event in ("-CONTROL_SELECT_DATASET-", "-HOME_LOAD_DATASET-"):
        selector = ControlSelectDataset(dataset_manager)
        selected_dataset = selector.show(window)

        if selected_dataset:
            show_dataset_view(selected_dataset)

        return True

    if event in ("-CONTROL_SELECT_MODEL-", "-HOME_LOAD_MODEL-"):
        selector = ControlSelectModel(model_manager)
        selected_model = selector.show(window)

        if selected_model:
            show_model_view(selected_model)

        return True

    if event == "-HOME_OPEN_LOGS-":
        page_manager.show("-PAGE_LOGS-")
        sidebar_manager.set_active("-PAGE_LOGS-")
        update_responsive_components(window)
        window.refresh()
        return True

    return False


# ------------------------------------------------------------
#  PAGE EVENTS
# ------------------------------------------------------------
def handle_active_page_event(event, values):

    active_handler = page_manager.get_active_page_handler()

    if active_handler and hasattr(active_handler, "handle_event"):
        handled = active_handler.handle_event(event, values, window)

        if handled:
            return True

    return False



def _canonical_dataset_role(dataset_name):
    ds = dataset_manager.get(dataset_name)
    role = str(getattr(ds, "role", "mixed") if ds else "mixed").lower()
    return {"prediction": "evaluation", "validation": "predictive", "ground_truth": "predictive",
            "survey": "evaluation", "discovery": "evaluation"}.get(role, role)

def _check_model_dataset_compatibility(model_name, dataset_name, purpose):
    compatibility = model_manager.check_dataset_compatibility(model_name, dataset_name)
    if not compatibility.get("compatible"):
        sg.popup_error(
            f"The selected {purpose} dataset does not match the model input channels.\n\n"
            f"Dataset profile: {compatibility.get('dataset_profile')}\n"
            f"Dataset channels: {compatibility.get('dataset_channels')}\n"
            f"Model profile: {compatibility.get('model_profile')}\n"
            f"Model channels: {compatibility.get('model_channels')}"
        )
        viewer_model.update_profile_status(compatibility)
        return None
    viewer_model.update_profile_status(compatibility)
    return compatibility

# ------------------------------------------------------------
#  TASK EVENTS
# ------------------------------------------------------------
def handle_task_event(event, values):

    if not isinstance(event, str) or not event.startswith("-TASK_"):
        return False

    if event == "-TASK_CREATE_DATASET-":
        try:
            dc = dataset_manager.create_dataset_from_values(values)
            show_dataset_view(dc.dataset_name)
        except Exception as exc:
            sg.popup_error(f"Could not create dataset:\n{exc}")
        return True

    if event == "-TASK_CREATE_MODEL-":
        try:
            mc = model_manager.create_model_from_values(values)
            show_model_view(mc.model_name)
        except Exception as exc:
            sg.popup_error(f"Could not create model:\n{exc}")
        return True

    if event == "-TASK_MODEL_SET_DATASET-":
        data = values[event]
        model_name = data["model_name"]
        dataset_name = data["dataset_name"]

        model_manager.set_training_dataset(model_name, dataset_name)
        model_manager.reload()
        cfg = model_manager.get(model_name)
        compatibility = model_manager.check_dataset_compatibility(model_name, dataset_name)

        viewer_model.load_model(cfg, window)
        viewer_model.update_profile_status(compatibility)
        viewer_model.apply_stage("training")

        return True

    if event == "-TASK_MODEL_SET_PREDICTION_DATASET-":
        # ---------------------------------------------------------------------
        # Backwards-compatible handler for older view_model versions.
        # New role-based UI keeps predictive/evaluation dataset selections local
        # to the page and passes them directly to the relevant task.
        # ---------------------------------------------------------------------
        data = values[event]
        model_name = data.get("model_name") if isinstance(data, dict) else None
        dataset_name = data.get("dataset_name") if isinstance(data, dict) else None

        if model_name and dataset_name:
            compatibility = model_manager.check_dataset_compatibility(model_name, dataset_name)
            viewer_model.update_profile_status(compatibility)

        return True

    if event == "-TASK_TRAIN_MODEL-":
        model_name = values[event]
        cfg = model_manager.get(model_name)
        dataset_name = cfg.training_dataset

        active_task_windows["train_model"] = PageCreateModelTask()
        active_task_windows["train_model"].open()

        task_manager.start_task("train_model",
                                target=train_model_task,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                model_manager=model_manager,
                                dataset_manager=dataset_manager,
        )

        return True

    if event in ("-TASK_RUN_PREDICTIVE_TEST-", "-TASK_RUN_PREDICTION-"):
        payload = values[event]

        if isinstance(payload, dict):
            model_name = payload.get("model_name")
            dataset_name = payload.get("dataset_name")
        else:
            model_name = payload
            dataset_name = None

        if not model_name or not dataset_name:
            sg.popup_error("Choose an evaluation/ground-truth dataset before running model evaluation.")
            return True

        role = _canonical_dataset_role(dataset_name)
        if role != "predictive":
            sg.popup_error(
                "Evaluate + Metrics requires a labelled evaluation/ground-truth dataset.\n\n"
                f"Selected dataset '{dataset_name}' has role: {role}"
            )
            return True

        if _check_model_dataset_compatibility(model_name, dataset_name, "evaluation") is None:
            return True

        active_task_windows["predictive_test"] = PageCreateModelTask()
        active_task_windows["predictive_test"].open()

        task_manager.start_task("predictive_test",
                                target=run_prediction_task,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                model_manager=model_manager,
                                dataset_manager=dataset_manager,
                                workflow="predictive",
        )

        return True

    if event == "-TASK_RUN_EVALUATION-":
        payload = values[event]
        model_name = payload.get("model_name") if isinstance(payload, dict) else None
        dataset_name = payload.get("dataset_name") if isinstance(payload, dict) else None

        if not model_name or not dataset_name:
            sg.popup_error("Choose a prediction/discovery dataset before finding anomalies.")
            return True

        role = _canonical_dataset_role(dataset_name)
        if role != "evaluation":
            sg.popup_error(
                "Find Anomalies requires a prediction/discovery dataset.\n\n"
                f"Selected dataset '{dataset_name}' has role: {role}"
            )
            return True

        if _check_model_dataset_compatibility(model_name, dataset_name, "evaluation") is None:
            return True

        active_task_windows["evaluation_discovery"] = PageCreateModelTask()
        active_task_windows["evaluation_discovery"].open()

        task_manager.start_task("evaluation_discovery",
                                target=run_prediction_task,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                model_manager=model_manager,
                                dataset_manager=dataset_manager,
                                workflow="evaluation",
        )

        return True

    if event == "-TASK_CALIBRATE_THRESHOLD-":
        payload = values[event]

        if not isinstance(payload, dict):
            sg.popup_error("Threshold calibration event did not include a task payload.")
            return True

        model_name = payload.get("model_name")
        dataset_name = payload.get("dataset_name")

        if not model_name or not dataset_name:
            sg.popup_error("Threshold calibration needs both model_name and dataset_name.")
            return True

        role = _canonical_dataset_role(dataset_name)
        if role != "predictive":
            sg.popup_error(
                "Threshold calibration requires a labelled evaluation/ground-truth dataset.\n\n"
                f"Selected dataset '{dataset_name}' has role: {role}"
            )
            return True

        compatibility = model_manager.check_dataset_compatibility(model_name, dataset_name)

        if not compatibility.get("compatible"):
            sg.popup_error("The selected calibration dataset does not match the model input profile.\n\n"
                           f"Dataset profile: {compatibility.get('dataset_profile')}\n"
                           f"Dataset channels: {compatibility.get('dataset_channels')}\n"
                           f"Model profile: {compatibility.get('model_profile')}\n"
                           f"Model channels: {compatibility.get('model_channels')}"
            )
            viewer_model.update_profile_status(compatibility)
            return True

        active_task_windows["calibrate_threshold"] = PageCreateModelTask()
        active_task_windows["calibrate_threshold"].open()

        task_manager.start_task("calibrate_threshold",
                                target=calibrate_prediction_threshold_task,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                model_manager=model_manager,
                                dataset_manager=dataset_manager,
        )

        return True

    if event == "-TASK_MODEL_VISUALISATION-":
        payload = values[event]

        if not isinstance(payload, dict):
            sg.popup_error("Model visualisation event did not include a task payload.")
            return True

        model_name = payload.get("model_name")
        visual_type = payload.get("visual_type")

        if not model_name or not visual_type:
            sg.popup_error("Model visualisation needs both model_name and visual_type.")
            return True

        if visual_type == "anomaly_map":
            task_id = "model_anomaly_map"

        elif visual_type == "clustering":
            task_id = "model_clustering"

        else:
            sg.popup_error(f"Unknown model visualisation type: {visual_type}")
            return True

        active_task_windows[task_id] = PageCreateModelTask()
        active_task_windows[task_id].open()

        task_manager.start_task(task_id,
                                target=model_visualisation_task,
                                model_name=model_name,
                                visual_type=visual_type,
                                model_manager=model_manager,
                                dataset_manager=dataset_manager,
        )

        return True

    if event == "-TASK_LOAD_DATASET-":
        task_manager.start_task("load_dataset", target=load_dataset_task)
        return True

    if event == "-TASK_PROCESS_DATASET-":

        dataset_name = values[event]
        active_task_windows["process_dataset"] = PageCreateDatasetTask()
        active_task_windows["process_dataset"].open()

        task_manager.start_task("process_dataset",
                                target=process_dataset_task,
                                dataset_name=dataset_name,
                                dataset_manager=dataset_manager,
        )

        return True

    if event == "-TASK_GENERATE_STATISTICS-":

        dataset_name = values[event]
        active_task_windows["generate_statistics"] = PageCreateDatasetTask()
        active_task_windows["generate_statistics"].open()

        task_manager.start_task("generate_statistics",
                                target=generate_statistics_task,
                                dataset_name=dataset_name,
                                dataset_manager=dataset_manager,
        )

        return True

    return False




def _find_task_window(event_window):

    for task_id, task_window in active_task_windows.items():
        if task_window is not None and getattr(task_window, "window", None) == event_window:
            return task_id, task_window

    return None, None


# ------------------------------------------------------------
#  TASK WINDOW EVENTS
# ------------------------------------------------------------
def handle_task_window_event(event_window, event):

    task_id, task_window = _find_task_window(event_window)

    if task_window is None:
        return False

    if event in (sg.WIN_CLOSED, "-CMT_CANCEL-", "-CDT_CANCEL-"):
        choice = sg.popup_yes_no(
            "Cancel the current task?\n\n"
            "The task will stop at the next safe cancellation point.",
            title="Cancel Task",
            keep_on_top=True,
        )

        if choice == "Yes":
            task_manager.cancel_task(task_id)

            if hasattr(task_window, "mark_cancelling"):
                task_window.mark_cancelling()

        return True

    return True

# ------------------------------------------------------------
#  WORKER MESSAGES
# ------------------------------------------------------------
def handle_worker_messages():

    for task_id, msg_type, data in task_manager.get_messages():

        task_window = active_task_windows.get(task_id)

        if task_window is not None:
            task_window.on_worker_message(task_id, msg_type, data)

            if msg_type == "result" and isinstance(data, dict) and data.get("output_path"):

                ImageUtility().show_image_window(data["output_path"],
                                                 title=data.get("title", "Model Visualisation"),
                )

            if msg_type == "result" and task_id in {"run_prediction", "predictive_test", "evaluation_discovery"} and isinstance(data, dict):

                workflow_name = "Evaluation" if data.get("workflow") == "predictive" else "Prediction"
                sg.popup_scrolled(f"{workflow_name} completed.\n\n"
                                  f"Dataset: {data.get('dataset')}\n"
                                  f"Tiles: {data.get('tile_count')}\n"
                                  f"Outputs: {data.get('output_dir')}\n"
                                  f"Summary: {data.get('summary_path')}",
                                  title=f"{workflow_name} Complete",
                                  size=(90, 12),
                )

            
            if msg_type == "result" and isinstance(data, dict):

                if data.get("show_view") == "dataset" and data.get("dataset"):
                    try:
                        show_dataset_view(data["dataset"])

                    except Exception as exc:
                        sg.popup_error(f"Could not refresh dataset view:\n{exc}")

                
                if data.get("show_view") == "model" and data.get("model"):
                    
                    try:
                        stage = "trained" if task_id == "train_model" else None
                        show_model_view(data["model"], stage=stage)

                    except Exception as exc:
                        sg.popup_error(f"Could not refresh model view:\n{exc}")

            if msg_type == "finished":
                active_task_windows[task_id] = None

            continue

        active_handler = page_manager.get_active_page_handler()

        if active_handler and hasattr(active_handler, "on_worker_message"):
            active_handler.on_worker_message(task_id, msg_type, data)


# ------------------------------------------------------------
#  EVENT LOOP
# ------------------------------------------------------------
while True:

    event_window, event, values = sg.read_all_windows(timeout=50)

    if event_window == window and event == sg.WIN_CLOSED:
        break

    if event_window is not None and event_window != window:
        if handle_task_window_event(event_window, event):
            handle_worker_messages()
            continue

    if handle_control_event(event, values):
        continue

    if event in page_manager.pages:
        page_manager.show(event)
        sidebar_manager.set_active(event)

        if event == "-PAGE_DATASETS-":
            page_datasets.refresh(window)
        elif event == "-PAGE_MODELS-":
            page_models.refresh(window)

        update_responsive_components(window)
        window.refresh()
        continue

    if handle_active_page_event(event, values):
        continue

    if handle_task_event(event, values):
        continue

    handle_worker_messages()

    if event == "-WINDOW-RESIZED-":
        update_responsive_components(window)

window.close()
