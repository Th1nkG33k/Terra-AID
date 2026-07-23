
import time
import math
import json
import psutil
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from Core.Managers.path_manager import PathManager


# ============================================================
# TRAINING MANAGER
#
# Handles MAE training with full reporting, early stopping,
# and optional threaded progress updates.
# ============================================================
class TrainingManager:

    def __init__(self):
        self.paths = PathManager()

    # ---------------------------------------------------------
    # Metric helpers
    # ---------------------------------------------------------
    @staticmethod
    def mse_per_channel(recon, target):

        diff = (recon - target) ** 2
        return diff.mean(dim=[0, 2, 3]).detach().cpu().tolist()

    @staticmethod
    def psnr_from_mse(mse):

        if mse <= 0:
            return 100.0
        
        return 20 * math.log10(1.0) - 10 * math.log10(mse)

# --------------------------------------------------------------------------
# Normalise outputs from MAE, CNN AE, and ResNet AE models.
# --------------------------------------------------------------------------
    @staticmethod
    def unpack_reconstruction(output):
        
        if isinstance(output, tuple):
            recon = output[0]
            meta = output[1] if len(output) > 1 else {}

        else:
            recon = output
            meta = {}

        return recon, meta


# --------------------------------------------------------------------------
# Read from dict configs or SimpleNamespace-style configs.
# --------------------------------------------------------------------------
    @staticmethod
    def _cfg_get(config, section, key=None, default=None):
        
        section_obj = config.get(section, {}) if isinstance(config, dict) else getattr(config, section, {})
        
        if key is None:
            return section_obj
        
        if isinstance(section_obj, dict):
            return section_obj.get(key, default)
        
        return getattr(section_obj, key, default)

    # ---------------------------------------------------------
    # Training function
    #    Train any reconstruction-based anomaly model with full reporting.
    #    Supported architectures include MAE, Classic CNN Autoencoder, and
    #    ResNet Autoencoder as long as they return recon or (recon, meta).
    # ---------------------------------------------------------
    def train_reconstruction_model(self, model, train_dataset, val_dataset, config, save_dir, device="cpu", worker=None):

        device = torch.device(device)

        # -----------------------------------------------------
        # Setup
        # -----------------------------------------------------
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]
        lr = config["optimizer"]["lr"]
        weight_decay = config["optimizer"]["weight_decay"]
        patience = config["training"]["early_stopping_patience"]
        num_workers = config["training"]["num_workers"]
        model_name = config["model"]["name"]

        model = model.to(device)
        mse_loss_fn = nn.MSELoss()
        mae_loss_fn = nn.L1Loss()

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay,
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
        )

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
        )

        # -----------------------------------------------------
        # Early stopping
        # -----------------------------------------------------
        best_val_loss = float("inf")
        patience_counter = 0
        training_log = []

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------
        # Training loop
        # -----------------------------------------------------
        for epoch in range(epochs):

            epoch_start = time.time()

            if worker:
                worker.status(f"Epoch {epoch+1}/{epochs} started")

            model.train()
            train_mse = 0.0
            train_mae = 0.0
            train_psnr = 0.0

            num_batches = len(train_loader)

            # -------------------------
            # Training batches
            # -------------------------
            for batch_idx, (x, meta) in enumerate(train_loader):

                if worker and worker.cancel_flag:
                    return best_val_loss, training_log
                
                x = x.to(device)

                optimizer.zero_grad()
                recon, meta = self.unpack_reconstruction(model(x))

                mse = mse_loss_fn(recon, x)
                mae = mae_loss_fn(recon, x)

                mse.backward()
                optimizer.step()

                train_mse += mse.item()
                train_mae += mae.item()
                train_psnr += self.psnr_from_mse(mse.item())

                # -------------------------------------------------
                # PROGRESS UPDATE (batch-level)
                # -------------------------------------------------
                if worker:

                    pct = int(((epoch + (batch_idx + 1) / num_batches) / epochs) * 100)
                    worker.progress(pct)
                    worker.status(f"Epoch {epoch+1}/{epochs} — Batch {batch_idx+1}/{num_batches}")

            train_mse /= len(train_loader)
            train_mae /= len(train_loader)
            train_psnr /= len(train_loader)

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------
            model.eval()
            val_mse = 0.0
            val_mae = 0.0
            val_psnr = 0.0
            val_channel_mse = None

            with torch.no_grad():

                for val_batch_idx, (x, meta) in enumerate(val_loader):

                    x = x.to(device)
                    recon, meta = self.unpack_reconstruction(model(x))

                    mse = mse_loss_fn(recon, x)
                    mae = mae_loss_fn(recon, x)

                    val_mse += mse.item()
                    val_mae += mae.item()
                    val_psnr += self.psnr_from_mse(mse.item())

                    if val_channel_mse is None:
                        val_channel_mse = self.mse_per_channel(recon, x)

            val_mse /= len(val_loader)
            val_mae /= len(val_loader)
            val_psnr /= len(val_loader)

            # -------------------------------------------------
            # Validation progress bump
            # -------------------------------------------------
            if worker:

                pct = int(((epoch + 0.99) / epochs) * 100)
                worker.progress(pct)
                worker.status(f"Epoch {epoch+1}/{epochs} validation complete")

            # -------------------------------------------------
            # Timing + Memory Reporting
            # -------------------------------------------------
            epoch_time = time.time() - epoch_start

            gpu_mem_alloc = None
            gpu_mem_reserved = None

            if device.type == "cuda":

                gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
                gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)

            cpu_mem_mb = psutil.Process().memory_info().rss / (1024**2)

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            log_entry = {"epoch": epoch + 1,
                         "train": {
                            "mse": train_mse,
                            "mae": train_mae,
                            "psnr": train_psnr,
                        },
                         "val": {
                            "mse": val_mse,
                            "mae": val_mae,
                            "psnr": val_psnr,
                            "per_channel_mse": val_channel_mse,
                        },
                        "system": {
                            "epoch_time_sec": epoch_time,
                            "cpu_memory_mb": cpu_mem_mb,
                            "gpu_memory_alloc_mb": gpu_mem_alloc,
                            "gpu_memory_reserved_mb": gpu_mem_reserved,
                        },
            }

            training_log.append(log_entry)

            # -------------------------------------------------
            # Checkpointing + early stopping
            # -------------------------------------------------

            # Always save latest epoch separately, so we never confuse it with best.
            latest_path = save_dir / f"{model_name}_latest.pt"
            torch.save(model.state_dict(), latest_path)

            # Optional: save every epoch for later analysis.
            epoch_path = save_dir / f"{model_name}_epoch_{epoch + 1:03d}.pt"
            torch.save(model.state_dict(), epoch_path)

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0

                # App default checkpoint = best checkpoint.
                best_path = save_dir / f"{model_name}.pt"
                explicit_best_path = save_dir / f"{model_name}_best.pt"

                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), explicit_best_path)

                if worker:
                    worker.status(
                        f"New best checkpoint saved at epoch {epoch + 1} "
                        f"(val MSE: {val_mse:.6f})"
                    )

            else:
                patience_counter += 1

                if worker:
                    worker.status(
                        f"No validation improvement "
                        f"({patience_counter}/{patience})"
                    )

                if patience_counter >= patience:
                    if worker:
                        worker.status(
                            f"Early stopping at epoch {epoch + 1}. "
                            f"Best val MSE: {best_val_loss:.6f}"
                        )
                    break

        return best_val_loss, training_log


# ----------------------------------------------------------------------------
# Backward-compatible wrapper for older call sites.
# ----------------------------------------------------------------------------
    def train_one_MAE_config(self, model, train_dataset, val_dataset, config, save_dir, device="cpu", worker=None):
        
        return self.train_reconstruction_model(model=model,
                                               train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               config=config,
                                               save_dir=save_dir,
                                               device=device,
                                               worker=worker,
        )


# =====================================================================
# EVALUATION MANAGER
# Evaluates a trained MAE model on a validation dataset.
# =====================================================================

class EvaluationManager:

    def __init__(self):
        pass


    def evaluate_reconstruction_model(self, model, val_dataset, config, device="cpu"):

        batch_size = config["training"]["batch_size"]
        num_workers = config["training"]["num_workers"]
        criterion = nn.MSELoss()

        loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
        )

        model = model.to(device)
        model.eval()

        recon_losses = []
        anomaly_scores = []

        with torch.no_grad():

            for batch in loader:

                if isinstance(batch, (tuple, list)):
                    x = batch[0]                    
                else:
                    x = batch

                x = x.to(device)
                recon, meta = TrainingManager.unpack_reconstruction(model(x))

                # Reconstruction loss
                loss = criterion(recon, x)
                recon_losses.append(loss.item())

                # Anomaly score. MAE exposes a visible-pixel mask; CNN/ResNet
                # models do not, so fall back to full reconstruction error.
                diff = torch.abs(recon - x)
                mask = meta.get("mask") if isinstance(meta, dict) else meta

                if torch.is_tensor(mask):
                    score = (diff * (1 - mask)).mean()
                else:
                    score = diff.mean()
                
                anomaly_scores.append(score.item())

        recon_loss = float(np.mean(recon_losses))
        anomaly_score = float(np.mean(anomaly_scores))
        composite = recon_loss + anomaly_score

        return {"recon_loss": recon_loss,
                "anomaly_score": anomaly_score,
                "composite_score": composite,
        }

# --------------------------------------------------------------------------
# Backward-compatible wrapper for older call sites.
# --------------------------------------------------------------------------
    def evaluate_MAE_model(self, model, val_dataset, config, device="cpu"):
        return self.evaluate_reconstruction_model(model, val_dataset, config, device=device)
