import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_logits(path):
    """Load logits from .pt or .npy files."""
    if path.endswith(".pt"):
        logits = torch.load(path, map_location="cpu")
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()
    elif path.endswith(".npy"):
        logits = np.load(path, allow_pickle=True)
        if isinstance(logits, tuple) or len(logits.shape) == 1:
            logits = logits[0]
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return logits.astype(np.float32)


def compute_msp_scores(logits):
    """Compute Maximum Softmax Probability (MSP) for each sample."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values.numpy()


def compute_auroc(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    return roc_auc_score(labels, scores) * 100


def compute_fpr95(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx] * 100


def main(yaml_path):
    """
    Maximum Softmax Probability (MSP) based OOD detection.

    Args:
        yaml_path: Path to configuration file
    """
    cfg = load_config(yaml_path)

    id_path = Path(cfg["id_dataset"]["logits"])
    ood_entries = cfg["ood_datasets"]

    print("=" * 60)
    print(f"MSP OOD Detection | Model: {cfg['model_name']}")
    print(f"ID dataset: {cfg['id_dataset']['name']}")
    print("=" * 60)

    id_logits = load_logits(str(id_path))
    id_msp = compute_msp_scores(id_logits)
    print(f"✓ Loaded ID logits: {id_logits.shape} | MSP mean={id_msp.mean():.4f}")

    results_table = []

    for ood_entry in ood_entries:
        ood_name = ood_entry["name"]
        ood_path = ood_entry["logits"]
        if not ood_path:
            continue

        print(f"→ Evaluating {ood_name} ...", end="", flush=True)
        ood_logits = load_logits(str(ood_path))
        ood_msp = compute_msp_scores(ood_logits)

        auroc = compute_auroc(id_msp, ood_msp)
        fpr95 = compute_fpr95(id_msp, ood_msp)

        results_table.append({
            "dataset": ood_name,
            "auroc": auroc,
            "fpr95": fpr95,
        })
        print(" done.")

    print("\n" + "=" * 60)
    print("SUMMARY: MSP OOD Results")
    print("=" * 60)
    print(f"{'OOD Dataset':<15} {'FPR95':<10} {'AUROC':<10} ")
    print("-" * 35)
    for r in results_table:
        print(f"{r['dataset']:<15} {r['fpr95']:<10.2f} {r['auroc']:<10.2f}")
    print("=" * 60)
    print("✓ MSP OOD Evaluation Complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MSP OOD detection using logits from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
