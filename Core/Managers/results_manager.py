
import json
import csv
import yaml

from pathlib import Path


# ========================================================================

# ========================================================================

class ResultshManager:

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Save a single result entry to a master log file
    # ---------------------------------------------------------
    def append_to_master_log(save_root, result_entry):

        log_path = Path(save_root) / "all_results.yaml"

        # Load existing log if present
        if log_path.exists():
            with open(log_path, "r") as f:
                log = yaml.safe_load(f) or []
        else:
            log = []

        log.append(result_entry)

        with open(log_path, "w") as f:
            yaml.dump(log, f)


    # ---------------------------------------------------------
    # Save CSV summary for quick comparison
    # ---------------------------------------------------------
    def write_csv_summary(save_root, results):

        csv_path = Path(save_root) / "summary.csv"

        fieldnames = ["config_id",
                      "encoder_depth",
                      "decoder_depth",
                      "base_channels",
                      "embed_dim",
                      "mask_ratio",
                      "recon_loss",
                      "anomaly_score",
                      "composite_score"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                cfg = r["config"]
                met = r["metrics"]

                writer.writerow({
                                "config_id": r["config_id"],
                                "encoder_depth": cfg.get("encoder_depth"),
                                "decoder_depth": cfg.get("decoder_depth"),
                                "base_channels": cfg.get("base_channels"),
                                "embed_dim": cfg.get("embed_dim"),
                                "mask_ratio": cfg.get("mask_ratio"),
                                "recon_loss": met["recon_loss"],
                                "anomaly_score": met["anomaly_score"],
                                "composite_score": met["composite_score"],
                })


    # ---------------------------------------------------------
    # Save JSON summary for UI integration
    # ---------------------------------------------------------
    def write_json_summary(save_root, best, results):

        json_path = Path(save_root) / "summary.json"

        summary = {
            "best_config": best,
            "all_results": results
        }

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)


    # ---------------------------------------------------------
    # Pretty print summary to console
    # ---------------------------------------------------------
    def print_summary(best):
        
        print("\n" + "=" * 60)
        print(" BEST ARCHITECTURE SUMMARY")
        print("=" * 60)

        cfg = best["config"]
        met = best["metrics"]

        print(f"Encoder depth:   {cfg['encoder_depth']}")
        print(f"Decoder depth:   {cfg['decoder_depth']}")
        print(f"Base channels:   {cfg['base_channels']}")
        print(f"Embed dim:       {cfg['embed_dim']}")
        print(f"Mask ratio:      {cfg['mask_ratio']}")
        print("-" * 60)
        print(f"Recon loss:      {met['recon_loss']:.6f}")
        print(f"Anomaly score:   {met['anomaly_score']:.6f}")
        print(f"Composite score: {met['composite_score']:.6f}")
        print("=" * 60)
