import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe in threads

import matplotlib.pyplot as plt
import json
import numpy as np

from pathlib import Path
from umap import UMAP

# ============================================================================
#    Dataset statistics generator for Terra-AId.
#
#    Loads profile-defined model_input.tif tiles from cfg.processed_path.
#    Computes:
#        - Band histograms
#        - Correlation matrix + heatmap
#        - UMAP (sampled automatically)
# ============================================================================
class StatisticsProcessor:

    def __init__(self, cfg):

        self.cfg = cfg
        self.root = Path(cfg.processed_path)
        self.pattern = getattr(cfg, "tile_folder_pattern", "tile *")

        print(f"[StatisticsProcessor] root={self.root} pattern='{self.pattern}'")

        # Collect tile folders
        self.tile_dirs = sorted(
            d for d in self.root.glob(self.pattern) if d.is_dir()
        )

        if not self.tile_dirs:
            raise RuntimeError(f"No tile folders found using pattern '{self.pattern}' in {self.root}")

    
    def run(self):

        stacks = self._load_model_input_stacks()

        if not stacks:
            raise RuntimeError("No model_input.tif stacks found — cannot compute statistics.")

        self._compute_histograms(stacks)
        self._compute_correlation(stacks)
        self._compute_umap(stacks)

    
    def _load_model_input_stacks(self):

        stacks = []
        
        for tile_dir in self.tile_dirs:
            model_input_path = tile_dir / "model_input.tif"

            if not model_input_path.exists():
                print(f"[WARN] Missing model_input.tif in {tile_dir}")
                continue

            import rasterio
            with rasterio.open(model_input_path) as src:
                arr = src.read().astype(np.float32)

            stacks.append(arr)

        return stacks

    
    def _compute_histograms(self, stacks):

        import matplotlib.pyplot as plt

        band_count = stacks[0].shape[0]
        fig, axes = plt.subplots(band_count, 1, figsize=(8, 2 * band_count))

        flat = [s.reshape(band_count, -1) for s in stacks]
        flat = np.hstack(flat)

        for i in range(band_count):
            axes[i].hist(flat[i], bins=100, color="steelblue")
            axes[i].set_title(f"Band {i}")

        out = self.cfg.paths.visuals_dir / f"{self.cfg.dataset_name}_band_histograms.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    
    def _compute_correlation(self, stacks):

        band_count = stacks[0].shape[0]

        flat = [s.reshape(s.shape[0], -1) for s in stacks]
        flat = np.hstack(flat)

        # Replace NaN and Inf values before computing
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute correlation safely
        corr = np.corrcoef(flat)

        # Guard against degenerate results
        if np.isnan(corr).all() or np.isinf(corr).any():
            print("[WARN] Correlation matrix invalid — check input arrays.")
            corr = np.zeros((flat.shape[0], flat.shape[0]))


        # Save JSON
        out_json = self.cfg.paths.visuals_dir / f"{self.cfg.dataset_name}_correlation_matrix.json"
        out_json.write_text(json.dumps(corr.tolist(), indent=2))

        # Save heatmap
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(corr, cmap="viridis")
        fig.colorbar(cax)
        ax.set_title("Correlation Matrix")

        out_png = self.cfg.paths.visuals_dir / f"{self.cfg.dataset_name}_correlation_heatmap.png"
        fig.savefig(out_png)
        plt.close(fig)

    
    def _compute_umap(self, stacks, max_samples=200_000):
        
        band_count = stacks[0].shape[0]

        flat = [s.reshape(band_count, -1).T for s in stacks]
        X = np.vstack(flat)

        total = X.shape[0]
        if total > max_samples:
            idx = np.random.choice(total, max_samples, replace=False)
            X = X[idx]
            print(f"[UMAP] Sampling {total:,} → {X.shape[0]:,}")

        X = np.nan_to_num(X)

        reducer = UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)

        # Save embedding
        out_npy = self.cfg.paths.visuals_dir / f"{self.cfg.dataset_name}_umap_raw_embedding.npy"
        np.save(out_npy, embedding)

        # Save plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, alpha=0.5)
        ax.set_title("UMAP (Raw Spectral)")

        out_png = self.cfg.paths.visuals_dir / f"{self.cfg.dataset_name}_umap_raw.png"
        fig.savefig(out_png)
        plt.close(fig)
