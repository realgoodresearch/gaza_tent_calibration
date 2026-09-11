"""Classify every valid pixel in the pre-war Gaza mosaic by land-use class."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent
INPUT_TIF = REPO_ROOT / "data/tif_files/prewar_gaza.tif"
MODEL_PATH = MODEL_DIR / "land_use_kmeans.pkl"
LABELS_PATH = MODEL_DIR / "manual_land_use_labels.json"
OUTPUT_TIF = MODEL_DIR / "prewar_gaza_land_use.tif"


def main() -> None:
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    with LABELS_PATH.open(encoding="utf-8") as file:
        labels = {int(key): str(value) for key, value in json.load(file).items()}

    with rasterio.open(INPUT_TIF) as source:
        if source.count < 3:
            raise ValueError(f"Expected at least three RGB bands in {INPUT_TIF}")

        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
            BIGTIFF="YES",
        )

        with rasterio.open(OUTPUT_TIF, "w", **profile) as destination:
            destination.set_band_description(1, "Land-use class")
            destination.update_tags(
                source=str(INPUT_TIF.relative_to(REPO_ROOT)),
                model=str(MODEL_PATH.relative_to(REPO_ROOT)),
                class_values="Model classes are stored as class_id + 1; 0 is nodata",
                **{f"class_{class_id + 1}": label for class_id, label in labels.items()},
            )

            for _, window in source.block_windows(1):
                rgb = source.read((1, 2, 3), window=window).transpose(1, 2, 0)
                if source.count >= 4:
                    valid = source.read(4, window=window) > 0
                else:
                    valid = source.dataset_mask(window=window) > 0

                classified = np.zeros(valid.shape, dtype=np.uint8)
                pixels = rgb[valid].astype(np.float32, copy=False)
                if pixels.size:
                    squared_distance = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                    classified[valid] = np.argmin(squared_distance, axis=1).astype(np.uint8) + 1
                destination.write(classified, 1, window=window)

    print(f"Saved classified land-use raster to {OUTPUT_TIF}")


if __name__ == "__main__":
    main()
