from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = Path.cwd()

H5_PATH = REPO_ROOT / "validation" / "tile_validation_samples.h5"
MODEL_DIR = REPO_ROOT / "validation" / "land_use_classification"
MODEL_PATH = MODEL_DIR / "land_use_kmeans.pkl"
LABELS_PATH = MODEL_DIR / "manual_land_use_labels.json"
EXAMPLE_PATH = MODEL_DIR / "land_use_examples.png"

N_CLUSTERS = 3
MAX_PIXELS_PER_TILE = 15000
MAX_TRAINING_PIXELS = 200000
RANDOM_STATE = 42


def ensure_label_file() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    expected_labels = {str(i): "" for i in range(N_CLUSTERS)}

    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing_keys = {str(key) for key in existing.keys()}
                expected_keys = set(expected_labels.keys())
                if existing_keys == expected_keys:
                    return

                merged = {key: str(existing.get(key, "")) for key in expected_labels.keys()}
                with open(LABELS_PATH, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2, sort_keys=True)
                return
        except (json.JSONDecodeError, OSError):
            pass

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(expected_labels, f, indent=2, sort_keys=True)


def load_prewar_training_pixels(
    h5_path: Path,
    max_pixels_per_tile: int = MAX_PIXELS_PER_TILE,
    max_total_pixels: int = MAX_TRAINING_PIXELS,
) -> np.ndarray:
    if not h5_path.exists():
        raise FileNotFoundError(f"Validation HDF5 not found: {h5_path}")

    pixels: List[np.ndarray] = []
    total = 0

    with h5py.File(h5_path, "r") as h5:
        sample_names = sorted(h5.keys(), key=lambda name: int(name.replace("sample_", "")))

        for sample_name in sample_names:
            group = h5[sample_name]
            if "prewar_rgb" not in group:
                continue

            image = np.asarray(group["prewar_rgb"], dtype=np.float32)
            flat = image.reshape(-1, 3)

            if flat.shape[0] > max_pixels_per_tile:
                rng = np.random.default_rng(42 + total)
                idx = rng.choice(flat.shape[0], size=max_pixels_per_tile, replace=False)
                flat = flat[idx]

            pixels.append(flat)
            total += flat.shape[0]

            if total >= max_total_pixels:
                break

    if not pixels:
        raise ValueError(f"No prewar RGB tiles found in {h5_path}")

    training = np.vstack(pixels)
    if training.shape[0] > max_total_pixels:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(training.shape[0], size=max_total_pixels, replace=False)
        training = training[idx]

    return training


def train_land_use_model(
    h5_path: Path = H5_PATH,
    n_clusters: int = N_CLUSTERS,
) -> KMeans:
    ensure_label_file()
    training_pixels = load_prewar_training_pixels(h5_path)

    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=RANDOM_STATE,
        max_iter=300,
    )
    model.fit(training_pixels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def predict_majority_class(prewar_rgb: np.ndarray, model: KMeans) -> int:
    flat = np.asarray(prewar_rgb, dtype=np.float32).reshape(-1, 3)
    labels = model.predict(flat)
    counts = np.bincount(labels)
    return int(np.argmax(counts))


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert float or uint8 RGB arrays into a valid imshow display array."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)

    if arr.max() <= 1.0:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)

    return arr.astype(np.uint8)


def save_example_grid(
    model: KMeans,
    h5_path: Path = H5_PATH,
    output_path: Path = EXAMPLE_PATH,
    n_samples: int = 20,
) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"Validation HDF5 not found: {h5_path}")

    sample_names = []
    with h5py.File(h5_path, "r") as h5:
        sample_names = sorted(
            h5.keys(), key=lambda name: int(name.replace("sample_", ""))
        )

    chosen = sample_names[:n_samples]
    n_rows = max(1, len(chosen))
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3 * n_rows))
    if n_rows == 1:
        axes = np.array([axes], dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.get_cmap("tab10", model.n_clusters)
    legend_handles = [
        plt.matplotlib.patches.Patch(color=cmap(cluster_id)[:3], label=f"Class {cluster_id}")
        for cluster_id in range(model.n_clusters)
    ]

    with h5py.File(h5_path, "r") as h5:
        for row_idx, sample_name in enumerate(chosen):
            group = h5[sample_name]
            prewar = np.asarray(group["prewar_rgb"], dtype=np.float32)
            labels = model.predict(prewar.reshape(-1, 3)).reshape(prewar.shape[:2])

            colorized = np.zeros(prewar.shape, dtype=np.float32)
            for cluster_id in range(model.n_clusters):
                color = np.asarray(cmap(cluster_id)[:3], dtype=np.float32)
                colorized[labels == cluster_id] = color

            prewar_display = to_display_rgb(prewar)
            colorized_display = np.clip(colorized, 0.0, 1.0)

            axes[row_idx, 0].imshow(prewar_display)
            axes[row_idx, 0].set_title(f"Prewar {sample_name}")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(colorized_display)
            axes[row_idx, 1].set_title(f"Clustered {sample_name}")
            axes[row_idx, 1].axis("off")

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=model.n_clusters,
        frameon=False,
        title="Land-use cluster classes",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_saved_model(model_path: Path = MODEL_PATH) -> KMeans:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def main() -> None:
    ensure_label_file()
    model = train_land_use_model()
    save_example_grid(model)
    print(f"Saved k-means model to {MODEL_PATH}")
    print(f"Saved manual label template to {LABELS_PATH}")
    print(f"Saved example grid to {EXAMPLE_PATH}")


if __name__ == "__main__":
    main()
