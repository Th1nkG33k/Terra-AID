
# ============================================================
#    DATASET TASKS
#
#    Background worker functions for dataset workflows.
#    mainW.py starts these tasks, but does not own the work.
# ============================================================


# ---------------------------------------------------------
# Send a staged progress message if a worker is available.
# ---------------------------------------------------------
def _progress(worker, pct, message):

    if worker:
        worker.progress(f"{pct}% - {message}")


# ---------------------------------------------------------
# Placeholder dataset load task.
# ---------------------------------------------------------
def load_dataset_task(worker=None, **kwargs):

    _progress(worker, 10, "Loading dataset")
    _progress(worker, 100, "Dataset loaded")

    return {"message": "Dataset loaded."}


# ---------------------------------------------------------
# Run the dataset processing pipeline.
# ---------------------------------------------------------
def process_dataset_task(dataset_name, dataset_manager, worker=None, **kwargs):

    if worker:
        worker.status(f"Processing dataset {dataset_name}...")

    _progress(worker, 5, "Starting dataset processing")
    _progress(worker, 15, "Loading dataset config")

    dataset_manager.process_dataset(dataset_name)

    _progress(worker, 95, "Reloading processed dataset")
    dataset_manager.reload()
    _progress(worker, 100, "Dataset processing complete")

    return {"message": "Dataset processed successfully.",
            "dataset": dataset_name,
            "show_view": "dataset",
    }


# ---------------------------------------------------------
# Run dataset statistics generation.
# ---------------------------------------------------------
def generate_statistics_task(dataset_name, dataset_manager, worker=None, **kwargs):

    if worker:
        worker.status(f"Generating statistics for {dataset_name}...")

    _progress(worker, 5, "Starting statistics generation")
    _progress(worker, 25, "Loading processed tiles")

    dataset_manager.generate_statistics(dataset_name)

    _progress(worker, 95, "Reloading statistics config")
    dataset_manager.reload()
    _progress(worker, 100, "Statistics generation complete")

    return {"message": "Statistics generated successfully.",
            "dataset": dataset_name,
            "show_view": "dataset",
    }
