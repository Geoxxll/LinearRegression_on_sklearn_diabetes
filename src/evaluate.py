import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def mse(y_true, y_pred):
    return ((y_true.numpy() - y_pred.numpy()) ** 2).mean()

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

def plot_loss(history, out_path="results/loss_curve.png"):
    Path("results").mkdir(exist_ok=True)
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("Training Loss Curve (MSE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def scatter_pred(y_true, y_pred, out_path="results/pred_vs_true.png"):
    Path("results").mkdir(exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs True")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], lw=1, color="red")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()