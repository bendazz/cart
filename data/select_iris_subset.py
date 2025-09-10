import json
from pathlib import Path
from typing import List, Dict, Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "iris_subset.json"
RANDOM_SEED = 42
TOTAL_SAMPLES = 20


def create_iris_subset(total_samples: int = TOTAL_SAMPLES, random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Create a deterministic stratified subset of the Iris dataset.

    Returns a dict with keys: feature_names, target_names, samples (list of dicts)
    Each sample dict: {"features": [...], "target": int, "target_name": str, "index": original_index}
    """
    iris = load_iris(as_frame=True)
    X = iris.data.values
    y = iris.target.values

    # Determine per-class counts (roughly balanced)
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)

    base_per_class = total_samples // n_classes
    remainder = total_samples % n_classes

    per_class_counts = {cls: base_per_class for cls in unique}
    # distribute remainder deterministically by class id order
    for cls in unique[:remainder]:
        per_class_counts[cls] += 1

    # Collect indices per class
    rng = np.random.default_rng(random_seed)
    selected_indices: List[int] = []
    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        # deterministic shuffle
        rng.shuffle(cls_indices)
        take = per_class_counts[cls]
        selected_indices.extend(cls_indices[:take].tolist())

    # Sort selected indices to keep consistent ordering (optional)
    selected_indices.sort()

    samples = []
    for idx in selected_indices:
        features = X[idx].tolist()
        target = int(y[idx])
        samples.append({
            "index": int(idx),
            "features": features,
            "target": target,
            "target_name": iris.target_names[target]
        })

    return {
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "samples": samples,
        "total_samples": len(samples),
        "random_seed": random_seed
    }


def main():
    subset = create_iris_subset()
    OUTPUT_FILE.write_text(json.dumps(subset, indent=2))
    print(f"Wrote {subset['total_samples']} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
