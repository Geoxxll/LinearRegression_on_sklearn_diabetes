from sklearn.datasets import load_diabetes
from typing import Tuple, List
import numpy as np
import torch

def load_diabetes_tensor() -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    diabetes = load_diabetes()
    X = torch.tensor(diabetes.data, dtype=torch.float32) # type: ignore
    y = torch.tensor(diabetes.target, dtype=torch.float32) # type: ignore
    feature_names = diabetes.feature_names  # type: ignore
    return X, y, feature_names
