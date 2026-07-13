import json
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in worker threads

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from pathlib import Path


# ============================================================================
# Dataset statistics generator for Terra-AID.
#
# Loads processed model_input.tif tiles from cfg.processed_path and writes:
#   - band histograms
#   - correlation matrix + heatmap
#   - UMAP projection, with PCA fallback if umap-learn is unavailable
#
# The processor is deliberately tolerant of the recent folder refactors:
#   Dataset/tile 0/model_input.tif
#   Dataset/tile_0/model_input.tif
#   Dataset/<any tile folder>/model_input.tif
# ============================================================================
class StatisticsProcessor:

    def __init__(self, cfg, worker=None):
        self.cfg = cfg
        self.worker = worker
        self.root = Path(cfg.processed_path).resolve()
        self.pattern = getattr(cfg, "tile_folder_pattern", "tile*") or "tile*"

        paths = getattr(cfg, "paths", None)
        visuals_dir = getattr(paths, "visuals_dir", None) if paths is not None else None
        if visuals_dir is None:
            root_dir = getattr(paths, "root", self.root.parent) if paths is not None else self.root.parent
            visuals_dir = Path(root_dir) / "Visuals"

        self.visuals_dir = Path(visuals_dir).resolve()
        self.visuals_dir.mkdir(parents=True, exist_ok=True)

        # Keep statistics lightweight enough for full AOI datasets.
        self.max_statistics_samples = int(getattr(cfg, "max_statistics_samples", 250_000))
        self.max_umap_samples = int(getattr(cfg, "max_umap_samples", 50_000))
        self.rng = np.random.default_rng(42)

        print(f"[StatisticsProcessor] root={self.root} pattern='{self.pattern}'")
        print(f"[StatisticsProcessor] visuals={self.visuals_dir}")

        self.tile_dirs = self._discover_tile_dirs()
        if not self.tile_dirs:
            raise RuntimeError(
                "No processed tile folders containing model_input.tif were found.\n\n"
                f"Checked: {self.root}\n"
                "Expected examples:\n"
                "  Dataset/tile 0/model_input.tif\n"
                "  Dataset/tile_0/model_input.tif"
            )

    # ------------------------------------------------------------------
    def _status(self, message: str):
        print(f"[StatisticsProcessor] {message}")
        if self.worker is not None:
            try:
                self.worker.status(message)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _discover_tile_dirs(self):
        if not self.root.exists():
            raise RuntimeError(f"Processed dataset folder does not exist: {self.root}")

        # If a single-tile dataset was written directly into Dataset/, support it.
        if self._find_model_input(self.root) is not None:
            return [self.root]

        candidates = []

        # Preferred route: configured pattern, usually tile*.
        for d in self.root.glob(self.pattern):
            if d.is_dir() and self._find_model_input(d) is not None:
                candidates.append(d)

        # Fallback route: support any child folder containing model_input.tif.
        if not candidates:
            for p in self.root.rglob("*"):
                if p.is_file() and p.name.lower() == "model_input.tif":
                    candidates.append(p.parent)

        unique = {d.resolve(): d for d in candidates}
        return sorted(unique.values(), key=self._tile_sort_key)

    # ------------------------------------------------------------------
    def _tile_sort_key(self, path: Path):
        digits = "".join(ch for ch in path.name if ch.isdigit())
        return (0, int(digits)) if digits else (1, path.name.lower())

    # ------------------------------------------------------------------
    def _find_model_input(self, tile_dir: Path):
        direct = tile_dir / "model_input.tif"
        if direct.exists():
            return direct

        try:
            for p in tile_dir.iterdir():
                if p.is_file() and p.name.lower() == "model_input.tif":
                    return p
        except FileNotFoundError:
            return None

        return None

    # ------------------------------------------------------------------
    def _channel_labels(self, model_input_path: Path, band_count: int):
        labels = []

        try:
            with rasterio.open(model_input_path) as src:
                labels = [d for d in src.descriptions if d]
        except Exception:
            labels = []

        if len(labels) != band_count:
            labels = list(getattr(self.cfg, "input_channels", []) or [])

        if len(labels) != band_count:
            labels = [f"Band {i}" for i in range(band_count)]

        return labels

    # ------------------------------------------------------------------
    def run(self):
        self._status(f"Generating statistics from {len(self.tile_dirs)} processed tile folder(s).")

        samples, labels, skipped = self._sample_model_input_pixels(max_samples=self.max_statistics_samples)

        if samples.size == 0:
            raise RuntimeError(
                "No valid pixels could be sampled from model_input.tif files — cannot compute statistics."
            )

        self._compute_histograms(samples, labels)
        self._compute_correlation(samples, labels)
        self._compute_umap(samples, labels)
        self._write_summary(samples, labels, skipped)

        self._status("Statistics generation complete.")

    # ------------------------------------------------------------------
    def _sample_model_input_pixels(self, max_samples: int):
        tile_inputs = []
        skipped = []

        for tile_dir in self.tile_dirs:
            model_input_path = self._find_model_input(tile_dir)
            if model_input_path is None:
                skipped.append({"tile": tile_dir.name, "reason": "model_input.tif not found"})
                continue
            tile_inputs.append((tile_dir, model_input_path))

        if not tile_inputs:
            raise RuntimeError("No model_input.tif stacks found — cannot compute statistics.")

        per_tile_limit = max(1, int(np.ceil(max_samples / len(tile_inputs))))
        samples = []
        labels = None
        expected_band_count = None

        for tile_dir, model_input_path in tile_inputs:
            try:
                with rasterio.open(model_input_path) as src:
                    arr = src.read().astype(np.float32)  # (C, H, W)
            except Exception as exc:
                skipped.append({"tile": tile_dir.name, "reason": f"failed to read model_input.tif: {exc}"})
                continue

            if arr.ndim != 3 or arr.shape[0] == 0:
                skipped.append({"tile": tile_dir.name, "reason": f"invalid stack shape: {arr.shape}"})
                continue

            band_count = arr.shape[0]
            if expected_band_count is None:
                expected_band_count = band_count
                labels = self._channel_labels(model_input_path, band_count)
            elif band_count != expected_band_count:
                skipped.append({
                    "tile": tile_dir.name,
                    "reason": f"channel count mismatch: expected {expected_band_count}, found {band_count}",
                })
                continue

            flat = arr.reshape(band_count, -1).T  # (pixels, bands)
            finite_mask = np.all(np.isfinite(flat), axis=1)
            valid_idx = np.flatnonzero(finite_mask)

            if valid_idx.size == 0:
                # Fall back to finite replacement if a tile is mostly NaN/Inf.
                flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
                valid_idx = np.arange(flat.shape[0])

            sample_count = min(per_tile_limit, valid_idx.size)
            chosen = self.rng.choice(valid_idx, size=sample_count, replace=False)
            samples.append(flat[chosen])

        if not samples:
            return np.empty((0, 0), dtype=np.float32), labels or [], skipped

        X = np.vstack(samples).astype(np.float32, copy=False)

        if X.shape[0] > max_samples:
            chosen = self.rng.choice(X.shape[0], size=max_samples, replace=False)
            X = X[chosen]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, labels or [f"Band {i}" for i in range(X.shape[1])], skipped

    # ------------------------------------------------------------------
    def _compute_histograms(self, X: np.ndarray, labels: list[str]):
        self._status("Writing band histograms...")

        band_count = X.shape[1]
        fig, axes = plt.subplots(band_count, 1, figsize=(9, max(3, 2.1 * band_count)))
        axes = np.atleast_1d(axes)

        for i, ax in enumerate(axes):
            ax.hist(X[:, i], bins=100)
            ax.set_title(labels[i] if i < len(labels) else f"Band {i}")
            ax.set_ylabel("Pixels")

        axes[-1].set_xlabel("Value")
        out = self.visuals_dir / f"{self.cfg.dataset_name}_band_histograms.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def _compute_correlation(self, X: np.ndarray, labels: list[str]):
        self._status("Writing correlation matrix...")

        if X.shape[0] < 2:
            corr = np.eye(X.shape[1], dtype=np.float32)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr = np.corrcoef(X, rowvar=False)

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        out_json = self.visuals_dir / f"{self.cfg.dataset_name}_correlation_matrix.json"
        out_json.write_text(json.dumps(corr.tolist(), indent=2), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation Matrix")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        out_png = self.visuals_dir / f"{self.cfg.dataset_name}_correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def _install_pkg_resources_compat(self):
        """
        umap-learn 0.5.x may still import ``pkg_resources`` during import.
        Newer Python environments can have setuptools builds where that legacy
        module is missing, even though importlib.metadata is available.

        Rather than letting statistics generation fall back to PCA just because
        of that import-time compatibility issue, provide the tiny subset of
        pkg_resources that umap-learn normally needs for version discovery.
        """
        import sys
        import types

        if "pkg_resources" in sys.modules:
            return

        try:
            import pkg_resources  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        from importlib.metadata import PackageNotFoundError, version

        compat = types.ModuleType("pkg_resources")

        class DistributionNotFound(Exception):
            pass

        class VersionConflict(Exception):
            pass

        class Distribution:
            def __init__(self, project_name: str):
                self.project_name = project_name
                self.version = version(project_name)

        def get_distribution(project_name: str):
            try:
                return Distribution(project_name)
            except PackageNotFoundError as exc:
                raise DistributionNotFound(str(exc)) from exc

        compat.get_distribution = get_distribution
        compat.DistributionNotFound = DistributionNotFound
        compat.VersionConflict = VersionConflict

        sys.modules["pkg_resources"] = compat

    # ------------------------------------------------------------------
    def _compute_umap(self, X: np.ndarray, labels: list[str]):
        self._status("Writing UMAP projection...")

        if X.shape[0] < 3:
            raise RuntimeError("At least three valid pixels are required to generate a UMAP plot.")

        if X.shape[0] > self.max_umap_samples:
            chosen = self.rng.choice(X.shape[0], size=self.max_umap_samples, replace=False)
            X_umap = X[chosen]
        else:
            X_umap = X

        # Standardise per channel to stop high-range bands dominating the 2D projection.
        mean = X_umap.mean(axis=0, keepdims=True)
        std = X_umap.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        X_scaled = (X_umap - mean) / std

        method = "UMAP"
        fallback_reason = None
        try:
            self._install_pkg_resources_compat()
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_scaled)
        except Exception as exc:
            # UMAP failures are common when the dependency is missing or numba is
            # not installed correctly. Statistics should still complete, so fall
            # back to PCA and record the reason in the summary JSON. Keep the
            # plot title clean and put the technical detail in the summary file.
            method = "PCA fallback"
            fallback_reason = f"{exc.__class__.__name__}: {exc}"
            self._status(f"UMAP unavailable; using PCA fallback. Reason: {fallback_reason}")
            from sklearn.decomposition import PCA
            embedding = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

        out_npy = self.visuals_dir / f"{self.cfg.dataset_name}_umap_raw_embedding.npy"
        np.save(out_npy, embedding)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, alpha=0.5)
        ax.set_title(f"{method} (Raw Spectral)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        out_png = self.visuals_dir / f"{self.cfg.dataset_name}_umap_raw.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        self.umap_method = method
        self.umap_fallback_reason = fallback_reason
        self.umap_sample_count = int(X_scaled.shape[0])

    # ------------------------------------------------------------------
    def _write_summary(self, X: np.ndarray, labels: list[str], skipped: list[dict]):
        summary = {
            "dataset": self.cfg.dataset_name,
            "processed_root": str(self.root),
            "visuals_dir": str(self.visuals_dir),
            "tile_count_used": len(self.tile_dirs),
            "sample_count": int(X.shape[0]),
            "channel_count": int(X.shape[1]),
            "channels": labels,
            "umap_method": getattr(self, "umap_method", "unknown"),
            "umap_fallback_reason": getattr(self, "umap_fallback_reason", None),
            "umap_sample_count": getattr(self, "umap_sample_count", None),
            "skipped_tiles": skipped,
            "outputs": {
                "band_histograms": f"{self.cfg.dataset_name}_band_histograms.png",
                "correlation_matrix": f"{self.cfg.dataset_name}_correlation_matrix.json",
                "correlation_heatmap": f"{self.cfg.dataset_name}_correlation_heatmap.png",
                "umap_embedding": f"{self.cfg.dataset_name}_umap_raw_embedding.npy",
                "umap_plot": f"{self.cfg.dataset_name}_umap_raw.png",
            },
        }

        out_json = self.visuals_dir / f"{self.cfg.dataset_name}_statistics_summary.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
