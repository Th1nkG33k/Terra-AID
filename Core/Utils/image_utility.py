import io
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import PySimpleGUI as sg

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from Core.Managers.path_manager import PathManager


# ============================================================

# ============================================================

class ImageUtility:

    def __init__(self):
        self.paths = PathManager()

    # ------------------------------------------------------------
    # PATH + PNG HELPERS
    # ------------------------------------------------------------

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path

        # Prefer project-relative paths if they exist.
        project_path = (self.paths.PROJECT_ROOT / path).resolve()
        if project_path.exists():
            return project_path

        # Fall back to assets for legacy calls like "clear.png".
        return (self.paths.ASSETS_DIR / path).resolve()


    def fig_to_png_bytes(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()


    def _finish_fig(self, fig, save_path=None, show=False):

        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path

        if show:
            plt.show()
        else:
            plt.close(fig)

        return None

    def load_and_resize_image(self, path, width=None, height=None):

        img_path = self.resolve_path(path)

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            img = Image.open(img_path)

        except UnidentifiedImageError:
            raise ValueError(f"Invalid or corrupted image file: {img_path}")

        if width and not height:
            ratio = width / img.width
            height = int(img.height * ratio)

        elif height and not width:
            ratio = height / img.height
            width = int(img.width * ratio)

        if width and height:
            img = img.resize((int(width), int(height)), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        return buffer.getvalue()

    def show_image_window(self, path, title="Image Viewer", max_width=1000, max_height=700):

        try:
            img_path = self.resolve_path(path)
            with Image.open(img_path) as img:
                w, h = img.size

            scale = min(max_width / w, max_height / h, 1.0)
            width = int(w * scale)
            height = int(h * scale)
            png_bytes = self.load_and_resize_image(img_path, width=width, height=height)

        except Exception as e:
            sg.popup_error(f"Failed to load image:\n{e}")
            return

        layout = [
                    [sg.Image(data=png_bytes, key="-POPUP_IMAGE-")],
                    [sg.Push(), sg.Button("Close"), sg.Push()],
        ]

        win = sg.Window(title, layout, modal=False, resizable=True, finalize=True)

        while True:

            ev, _ = win.read()
            
            if ev in (sg.WIN_CLOSED, "Close"):
                break

        win.close()

    # ------------------------------------------------------------
    # DATASET VISUALS
    # ------------------------------------------------------------

    def compute_anomaly_map(self, original, reconstructed):
        error = (original - reconstructed) ** 2
        error = error.mean(dim=0).detach().cpu().numpy()
        error = error - error.min()
        if error.max() > 0:
            error = error / error.max()
        return error

    def unpack_reconstruction(self, output):

        if isinstance(output, tuple):
            recon = output[0]
            meta = output[1] if len(output) > 1 else {}

        else:
            recon = output
            meta = {}
        
        mask = meta.get("mask") if isinstance(meta, dict) else meta
        
        return recon, mask

    def colorize_anomaly_map(self, anomaly_map):

        anomaly_map = np.squeeze(anomaly_map)
        anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=1.0, neginf=0.0)
        anomaly_map = np.clip(anomaly_map, 0, 1)

        arr = (anomaly_map * 255).round().astype(np.uint8)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.applyColorMap(arr_bgr, cv2.COLORMAP_JET)

        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def overlay_heatmap_on_rgb(self, rgb, heatmap, alpha=0.5):

        heatmap_norm = heatmap.astype(np.float32) / 255.0
        overlay = (1 - alpha) * rgb + alpha * heatmap_norm
        
        return np.clip(overlay, 0, 1)

    def visualize_anomaly(self, original, reconstructed, save_path=None):

        anomaly = self.compute_anomaly_map(original, reconstructed)
        heatmap = self.colorize_anomaly_map(anomaly)

        rgb_orig = original[[3, 2, 1]].detach().cpu().numpy().transpose(1, 2, 0)
        rgb_recon = reconstructed[[3, 2, 1]].detach().cpu().numpy().transpose(1, 2, 0)

        rgb_orig = np.clip(rgb_orig, 0, 1)
        rgb_recon = np.clip(rgb_recon, 0, 1)
        overlay = self.overlay_heatmap_on_rgb(rgb_orig, heatmap)

        fig = plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1); plt.title("Original RGB"); plt.imshow(rgb_orig); plt.axis("off")
        plt.subplot(2, 2, 2); plt.title("Reconstructed RGB"); plt.imshow(rgb_recon); plt.axis("off")
        plt.subplot(2, 2, 3); plt.title("Anomaly Heatmap"); plt.imshow(heatmap); plt.axis("off")
        plt.subplot(2, 2, 4); plt.title("RGB + Anomaly Overlay"); plt.imshow(overlay); plt.axis("off")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)

    # ============================================================
    # MODEL VISUALS
    # ============================================================

    def _metric_series(self, training_log, section, metric):

        values = []
        
        for entry in training_log:

            # Current Terra-AId format: {"train": {"mse": ...}, "val": {"mse": ...}}
            if isinstance(entry.get(section), dict) and metric in entry[section]:
                values.append(entry[section][metric])
                continue

            # Legacy fallback: train_loss / val_loss.
            legacy_key = f"{section}_{metric}"

            if legacy_key in entry:
                values.append(entry[legacy_key])
                continue

            legacy_loss_key = f"{section}_loss"

            if metric == "mse" and legacy_loss_key in entry:
                values.append(entry[legacy_loss_key])
                continue

            values.append(np.nan)

        return values

    def plot_loss_curve(self, training_log, save_path=None):

        epochs = [e.get("epoch", i + 1) for i, e in enumerate(training_log)]
        train_loss = self._metric_series(training_log, "train", "mse")
        val_loss = self._metric_series(training_log, "val", "mse")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, label="Train MSE", marker="o")
        ax.plot(epochs, val_loss, label="Val MSE", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("Reconstruction Training Loss Curve")
        ax.legend()
        ax.grid(True)

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)

    def plot_training_metrics(self, training_log, save_path=None):

        epochs = [e.get("epoch", i + 1) for i, e in enumerate(training_log)]
        metrics = ["mse", "mae", "psnr"]

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        for ax, metric in zip(axs, metrics):
            ax.plot(epochs, self._metric_series(training_log, "train", metric), label=f"Train {metric.upper()}", marker="o")
            ax.plot(epochs, self._metric_series(training_log, "val", metric), label=f"Val {metric.upper()}", marker="o")
            ax.set_xlabel("Epoch")
            ax.set_title(metric.upper())
            ax.grid(True)
            ax.legend(fontsize=8)

        fig.suptitle("Training Metrics")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)

    def plot_per_channel_mse_from_log(self, training_log, channel_labels=None, save_path=None):

        rows = []
        labels = []

        for entry in training_log:

            per_channel = entry.get("val", {}).get("per_channel_mse")
            
            if per_channel:
                rows.append(per_channel)
                labels.append(f"Epoch {entry.get('epoch', len(labels) + 1)}")

        if not rows:
            raise ValueError("No val.per_channel_mse entries found in training_log.json")

        mse_matrix = np.array(rows, dtype=float)
        mse_matrix = np.log10(np.maximum(mse_matrix, 1e-12))

        if not channel_labels or len(channel_labels) != mse_matrix.shape[1]:
            channel_labels = [f"Ch {i}" for i in range(mse_matrix.shape[1])]

        return self.plot_per_channel_mse_heatmap(mse_matrix=mse_matrix,
                                                 channel_labels=channel_labels,
                                                 config_labels=labels,
                                                 save_path=save_path,
                                                 already_log_scaled=True,
        )

    def plot_reconstruction(self, input_tensor, recon_tensor, mask, save_path=None):

        inp = input_tensor.detach().cpu().numpy()
        rec = recon_tensor.detach().cpu().numpy()
        m = mask.detach().cpu().numpy()

        inp_rgb = np.stack([inp[3], inp[2], inp[1]], axis=-1)
        rec_rgb = np.stack([rec[3], rec[2], rec[1]], axis=-1)
        inp_rgb = np.clip(inp_rgb, 0, 1)
        rec_rgb = np.clip(rec_rgb, 0, 1)
        err = np.abs(inp_rgb - rec_rgb).mean(axis=-1)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(inp_rgb); axs[0].set_title("Input")
        axs[1].imshow(m[0], cmap="gray"); axs[1].set_title("Mask")
        axs[2].imshow(rec_rgb); axs[2].set_title("Reconstruction")
        axs[3].imshow(err, cmap="inferno"); axs[3].set_title("Error Heatmap")

        for ax in axs:
            ax.axis("off")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)

    def plot_per_channel_mse_heatmap(self, mse_matrix, channel_labels, config_labels, save_path=None, already_log_scaled=False):
        
        fig, ax = plt.subplots(figsize=(12, 6))
        matrix = np.array(mse_matrix, dtype=float)
        
        if not already_log_scaled:
            matrix = np.log10(np.maximum(matrix, 1e-12))

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="log10(MSE)")
        ax.set_xticks(range(len(channel_labels)))
        ax.set_xticklabels(channel_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(config_labels)))
        ax.set_yticklabels(config_labels)
        ax.set_title("Per-Channel Validation MSE Heatmap")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)

    def plot_val_mse_curve(self, results, save_path=None):

        fig, ax = plt.subplots(figsize=(10, 6))

        for label, vals in results.items():
            ax.plot(vals["epochs"], vals["mse"], label=label, marker="o")

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.set_title("Validation MSE Across Epochs")
        ax.legend()
        ax.grid(True)

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)


    # ------------------------------------------------------------
    # MODEL INFERENCE VISUALS
    #    Load the trained MAE checkpoint and the first tile from the model's
    #    training dataset. Returns (model, x, recon, mask, device).
    # ------------------------------------------------------------

    def _load_model_and_sample(self, model_cfg, dataset_cfg, worker=None, run_reconstruction=True):

        from Core.Pytorch.pytorch_dataset_factory import PyTorchDatasetFactory

        dataset = PyTorchDatasetFactory().build(dataset_cfg)

        if len(dataset) == 0:
            raise RuntimeError(f"Dataset '{dataset_cfg.dataset_name}' has no tiles in {dataset_cfg.processed_path}")

        x, meta = dataset[0]

        if x.ndim != 3:
            raise RuntimeError(f"Expected sample tensor shape (C,H,W), got {tuple(x.shape)}")

        # Keep generated previews lightweight. The MAE is fully convolutional,
        # so a smaller preview tile is valid and much faster for UI-triggered
        # visualisations.
        max_preview_size = 128

        if max(x.shape[1], x.shape[2]) > max_preview_size:

            x = torch.nn.functional.interpolate(x.unsqueeze(0),
                                                size=(max_preview_size, max_preview_size),
                                                mode="bilinear",
                                                align_corners=False,
            ).squeeze(0)

        # ModelConfig.build_model expects architecture.num_channels, but the
        # loaded config does not persist it. Infer it from the dataset sample.
        model_cfg.architecture.num_channels = int(x.shape[0])

        model = model_cfg.build_model()
        ckpt_path = Path(model_cfg.paths.checkpoints) / f"{model_cfg.model_name}.pt"

        if not ckpt_path.exists():
        
            matches = sorted(Path(model_cfg.paths.checkpoints).glob("*.pt"))
        
            if matches:
                ckpt_path = matches[0]
            else:
                raise FileNotFoundError(f"Checkpoint not found in {model_cfg.paths.checkpoints}")

        device = torch.device(getattr(model_cfg, "device", "cpu") or "cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        if worker:
            worker.status(f"Loading checkpoint: {ckpt_path.name}")

        state = torch.load(ckpt_path, map_location=device)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state)
        model.to(device)
        model.eval()

        if run_reconstruction:

            x_batch = x.unsqueeze(0).to(device)
            torch.manual_seed(0)

            with torch.no_grad():
                recon, mask = self.unpack_reconstruction(model(x_batch))
            
            recon = recon.squeeze(0).detach().cpu()
            
            if torch.is_tensor(mask):
                mask = mask.squeeze(0).detach().cpu()
            else:
                mask = torch.ones(1, x.shape[1], x.shape[2])
        else:
            recon = None
            mask = None

        return model, x.detach().cpu(), recon, mask, device


    # ------------------------------------------------------------
    # Return a display-safe RGB image from a C,H,W tensor.
    # ------------------------------------------------------------
    def _tensor_rgb(self, x):
        
        arr = x.detach().cpu().numpy()

        if arr.shape[0] >= 4:
            rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)
        
        elif arr.shape[0] >= 3:
            rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
        
        else:
            rgb = np.repeat(arr[0][..., None], 3, axis=-1)
        
        return np.clip(rgb, 0, 1)

    # -----------------------------------------------------------
    #  Generate and save a reconstruction-error anomaly visualisation for a
    #  trained MAE model. Existing callers expect the returned path.
    # -----------------------------------------------------------
    def generate_model_anomaly_map(self, model_cfg, dataset_cfg, save_dir, worker=None):

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "anomaly_map.png"

        if worker:
            worker.status("Running model reconstruction for anomaly map...")

        _, x, recon, mask, _ = self._load_model_and_sample(model_cfg, dataset_cfg, worker=worker)

        anomaly = self.compute_anomaly_map(x, recon)
        heatmap = self.colorize_anomaly_map(anomaly)
        rgb_orig = self._tensor_rgb(x)
        rgb_recon = self._tensor_rgb(recon)
        overlay = self.overlay_heatmap_on_rgb(rgb_orig, heatmap)

        fig = plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1); plt.title("Original RGB"); plt.imshow(rgb_orig); plt.axis("off")
        plt.subplot(2, 2, 2); plt.title("Reconstructed RGB"); plt.imshow(rgb_recon); plt.axis("off")
        plt.subplot(2, 2, 3); plt.title("Anomaly Heatmap"); plt.imshow(heatmap); plt.axis("off")
        plt.subplot(2, 2, 4); plt.title("RGB + Anomaly Overlay"); plt.imshow(overlay); plt.axis("off")

        return self._finish_fig(fig, save_path=save_path, show=False)

    # ------------------------------------------------------------------------
    #    Cluster feature rows quickly with OpenCV k-means. For large maps, fit
    #    the centres on a deterministic subset, then assign every pixel to its
    #    nearest centre. This keeps the UI responsive.
    # ------------------------------------------------------------------------
    def _kmeans_labels(self, features, n_clusters=6, max_fit_samples=20000):

        features = np.asarray(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        # Standardise for more stable clustering.
        std = features.std(axis=0, keepdims=True)
        features = (features - features.mean(axis=0, keepdims=True)) / np.maximum(std, 1e-6)

        if features.shape[0] > max_fit_samples:

            rng = np.random.default_rng(0)
            fit_idx = rng.choice(features.shape[0], size=max_fit_samples, replace=False)
            fit_features = features[fit_idx]
        
        else:
            fit_features = features

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.2)

        compactness, labels, centers = cv2.kmeans(fit_features,
                                                  int(n_clusters),
                                                  None,
                                                  criteria,
                                                  3,
                                                  cv2.KMEANS_PP_CENTERS,
        )

        # Assign every feature to its nearest centre in chunks to avoid a large
        # temporary distance matrix on bigger rasters.
        out = np.empty(features.shape[0], dtype=np.int32)
        chunk = 8192

        for start in range(0, features.shape[0], chunk):
            block = features[start:start + chunk]
            dists = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            out[start:start + chunk] = np.argmin(dists, axis=1)
        
        return out


    # ------------------------------------------------------------------
    #  Generate and save a pixel clustering map from the trained model's
    #  encoder features for the first tile of the model's training dataset.
    # ------------------------------------------------------------------
    def generate_model_clustering_map(self, model_cfg, dataset_cfg, save_dir, n_clusters=6, worker=None):

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "clustering_map.png"

        if worker:
            worker.status("Extracting encoder features for clustering...")

        model, x, recon, mask, device = self._load_model_and_sample(model_cfg, dataset_cfg, worker=worker, run_reconstruction=False)
        x_batch = x.unsqueeze(0).to(device)

        with torch.no_grad():

            if hasattr(model, "encode"):
                latent = model.encode(x_batch).squeeze(0).detach().cpu()
            else:
                latent = model.encoder(x_batch).squeeze(0).detach().cpu()

        lat = latent.numpy()
        c, h_lat, w_lat = lat.shape
        features = lat.reshape(c, -1).T

        if worker:
            worker.status(f"Clustering {features.shape[0]} feature pixels into {n_clusters} groups...")

        labels = self._kmeans_labels(features, n_clusters=n_clusters).reshape(h_lat, w_lat)

        # Upsample label map to the original tile size for display.
        h, w = x.shape[1], x.shape[2]
        labels_full = cv2.resize(labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(int)
        rgb_orig = self._tensor_rgb(x)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(rgb_orig)
        axs[0].set_title("Original RGB")
        axs[0].axis("off")
        im = axs[1].imshow(labels_full, cmap="tab20")
        axs[1].set_title(f"Encoder Clusters (k={n_clusters})")
        axs[1].axis("off")
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label="Cluster")

        return self._finish_fig(fig, save_path=save_path, show=False)

    # ------------------------------------------------------------
    # SEARCH VISUALS
    # ------------------------------------------------------------

    def plot_architecture_scores(self, results, save_path=None):
        ids = [r["config_id"] for r in results]
        scores = [r["metrics"]["composite_score"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(ids, scores)
        ax.set_xlabel("Config ID")
        ax.set_ylabel("Composite Score")
        ax.set_title("Architecture Search Results")
        ax.grid(axis="y")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)


    def plot_heatmap(self, results, x_key, y_key, save_path=None):

        xs = sorted(list(set(r["config"][x_key] for r in results)))
        ys = sorted(list(set(r["config"][y_key] for r in results)))

        heat = np.zeros((len(ys), len(xs)))

        for r in results:
            x = xs.index(r["config"][x_key])
            y = ys.index(r["config"][y_key])
            heat[y, x] = r["metrics"]["composite_score"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heat, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Composite Score")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"Architecture Heatmap: {y_key} vs {x_key}")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)



    def plot_search_convergence(self, trials, save_path=None):

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(trials, marker="o")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_title("Search Convergence Curve")
        ax.grid(True)

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)



    def plot_trial_comparison(self, scores, labels, save_path=None):
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, scores)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_title("Trial Comparison")
        ax.grid(axis="y")

        return self._finish_fig(fig, save_path=save_path, show=save_path is None)
