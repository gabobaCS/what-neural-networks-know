import yaml
import torch
import numpy as np
import faiss
import faiss.contrib.torch_utils
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-10)


def compute_knn_distance_faiss(test_acts_norm, index, k=1, batch_size=20000):
    """Compute KNN distances using FAISS with batching for memory efficiency."""
    test_acts_norm = np.ascontiguousarray(test_acts_norm.astype("float32"))
    all_distances = []
    for i in range(0, len(test_acts_norm), batch_size):
        batch = test_acts_norm[i:i + batch_size]
        distances, _ = index.search(batch, k)
        all_distances.append(distances)
    return np.vstack(all_distances)[:, -1]


def main(yaml_path, k=7):
    """
    KNN OOD detection using deep inversion features as reference bank.

    Args:
        yaml_path: Path to configuration file
        k: Number of nearest neighbors
    """
    cfg = load_config(yaml_path)

    id_path = Path(cfg["id_dataset"]["avgpool"])
    deep_inv_path = Path(cfg["deep_inversion"]["avgpool"])
    ood_entries = cfg["ood_datasets"]

    print("=" * 60)
    print(f"KNN OOD Detection (GPU-ready) | Deep Inversion Reference | Model: {cfg['model_name']}")
    print(f"ID dataset: {cfg['id_dataset']['name']}")
    print(f"K = {k}")
    print("=" * 60)

    id_data = torch.load(id_path)
    id_acts = id_data["activations"].numpy().astype("float32")
    print(f"✓ Loaded ID ({cfg['id_dataset']['name']}): {id_acts.shape}")

    deep_inv_data = torch.load(deep_inv_path)
    deep_inv_acts = deep_inv_data["activations"].numpy().astype("float32")
    print(f"✓ Loaded Deep Inversion reference: {deep_inv_acts.shape}")

    id_acts_norm = l2_normalize(id_acts)
    deep_inv_norm = l2_normalize(deep_inv_acts)

    d = deep_inv_norm.shape[1]
    cpu_index = faiss.IndexFlatL2(d)

    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(deep_inv_norm)
        index = gpu_index
        device = "GPU"
    else:
        cpu_index.add(deep_inv_norm)
        index = cpu_index
        device = "CPU"

    print(f"✓ Built FAISS index on {device} with {index.ntotal} Deep Inversion vectors (dim={d})")

    results_table = []

    for ood_entry in ood_entries:
        ood_name = ood_entry["name"]
        ood_path = ood_entry["avgpool"]
        if not ood_path:
            continue

        print(f"→ Evaluating {ood_name} ...", end="", flush=True)

        ood_data = torch.load(ood_path)
        ood_acts = ood_data["activations"].numpy().astype("float32")
        ood_acts_norm = l2_normalize(ood_acts)

        id_knn = compute_knn_distance_faiss(id_acts_norm, index, k=k)
        ood_knn = compute_knn_distance_faiss(ood_acts_norm, index, k=k)

        y_true = np.concatenate([np.ones(len(id_knn)), np.zeros(len(ood_knn))])
        y_scores = np.concatenate([-id_knn, -ood_knn])

        auroc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]

        results_table.append({
            "dataset": ood_name,
            "auroc": auroc,
            "fpr95": fpr95,
        })
        print(" done.")

    print("\n" + "=" * 60)
    print("SUMMARY: Deep Inversion Reference KNN OOD Results (GPU-ready)")
    print("=" * 60)
    print(f"{'OOD Dataset':<15} {'FPR95':<10} {'AUROC':<10}")
    print("-" * 35)
    for r in results_table:
        print(f"{r['dataset']:<15} {r['fpr95']:<10.4f} {r['auroc']:<10.4f} ")
    print("=" * 60)
    print(f"✓ Deep Inversion Reference Evaluation Complete (using {device})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GPU-ready KNN OOD detection using Deep Inversion as reference bank")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--k", type=int, default=7, help="Number of nearest neighbors")
    args = parser.parse_args()

    main(args.config, k=args.k)
