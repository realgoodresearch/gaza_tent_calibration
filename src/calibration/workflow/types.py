from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class FeatureBundle:
    frame: pd.DataFrame
    municipality_codes: np.ndarray
    municipality_names: List[str]
    design_matrix: np.ndarray
    design_columns: List[str]
    target: np.ndarray
    week_group: np.ndarray


@dataclass
class ModelSpec:
    name: str
    selected_candidates: List[str]
    final_columns: List[str]
    model_kind: str
    selected_candidates_count: List[str] | None = None
    selected_candidates_zero: List[str] | None = None
    count_columns: List[str] | None = None
    zero_columns: List[str] | None = None
