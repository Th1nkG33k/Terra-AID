from Core.Pytorch.pytorch_manager import MultimodalTileDataset


# ==================================================================
#    Builds PyTorch datasets from DatasetConfig objects.
#    Keeps PyTorchDataset clean and config‑agnostic.
# ==================================================================
class PyTorchDatasetFactory:

    def __init__(self):
        pass


    def build(self, dataset_config, transform=None):

        return MultimodalTileDataset(root_dir=dataset_config.processed_path,
                                     cfg=dataset_config,                     # ← ADD THIS
                                     bands=dataset_config.bands,
                                     tile_size=dataset_config.cfg.get("tile_size", 256),
                                     transform=transform
        )
