from Core.Pytorch.pytorch_manager import MultimodalTileDataset


# ==============================================================
#    Thin wrapper around MultimodalTileDataset for UI use.
#    - One dataset instance per config.
#    - Indexed access per tile.
#    - Caches first sample per tile for fast re‑use in the viewer.
# ==============================================================
class TileViewManager:

    def __init__(self, dataset_cfg, transform=None):

        self.cfg = dataset_cfg
        self.dataset = MultimodalTileDataset(root_dir=dataset_cfg.processed_path,
                                             cfg=dataset_cfg,
                                             bands=dataset_cfg.bands.included,
                                             tile_size=dataset_cfg.cfg.get("tile_size", 256),
                                             transform=transform,
        )

        self._cache = {}   # idx -> (tensor, meta)

    @property
    def tile_count(self) -> int:
        return len(self.dataset)

    def get_tile_sample(self, idx: int):

        if idx in self._cache:
            return self._cache[idx]

        x, meta = self.dataset[idx]
        self._cache[idx] = (x, meta)

        return x, meta

    def get_tile_name(self, idx: int) -> str:
        
        # Use folder name from metadata if present, else index.
        _, meta = self.get_tile_sample(idx)
        return meta.get("tile_id", f"tile {idx}")
