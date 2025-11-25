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
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "logits" in data:
                logits = data["logits"]
            elif isinstance(next(iter(data.values())), torch.Tensor):
                logits = next(iter(data.values()))
            else:
                raise ValueError(f"Unsupported dict keys: {data.keys()}")
        elif isinstance(data, torch.Tensor):
            logits = data
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")
    elif path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if logits.shape[0] == 1000 and logits.shape[1] > 1000:
        logits = logits.T
    return logits


def get_energy_score(logits, temperature=1.0):
    """Compute energy scores as logsumexp(logits / T)."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    logits = logits / temperature
    return torch.logsumexp(logits, dim=1).numpy()


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


def main(yaml_path, temperature=1.0):
    """
    Energy-based OOD detection.

    Args:
        yaml_path: Path to configuration file
        temperature: Temperature scaling parameter
    """
    cfg = load_config(yaml_path)

    id_path = Path(cfg["id_dataset"]["logits"])
    ood_entries = cfg["ood_datasets"]

    print("=" * 60)
    print(f"Energy OOD Detection | Model: {cfg['model_name']}")
    print(f"ID dataset: {cfg['id_dataset']['name']}")
    print(f"Temperature: {temperature}")
    print("=" * 60)

    id_logits = load_logits(str(id_path))
    id_scores = get_energy_score(id_logits, temperature)
    print(f"✓ Loaded ID logits: {id_logits.shape} | Energy mean={id_scores.mean():.4f}")

    results_table = []

    for ood_entry in ood_entries:
        ood_name = ood_entry["name"]
        ood_path = ood_entry["logits"]
        if not ood_path:
            continue

        print(f"→ Evaluating {ood_name} ...", end="", flush=True)
        ood_logits = load_logits(str(ood_path))
        ood_scores = get_energy_score(ood_logits, temperature)

        fpr95 = compute_fpr95(id_scores, ood_scores)
        auroc = compute_auroc(id_scores, ood_scores)

        results_table.append({
            "dataset": ood_name,
            "fpr95": fpr95,
            "auroc": auroc,
        })
        print(" done.")

    print("\n" + "=" * 60)
    print("SUMMARY: Energy OOD Results")
    print("=" * 60)
    print(f"{'OOD Dataset':<15} {'FPR95':<10} {'AUROC':<10}")
    print("-" * 35)
    for r in results_table:
        print(f"{r['dataset']:<15} {r['fpr95']:<10.2f} {r['auroc']:<10.2f}")
    print("=" * 60)
    print("✓ Energy OOD Evaluation Complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Energy OOD detection using logits from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling parameter")
    args = parser.parse_args()

    main(args.config, temperature=args.temperature)
