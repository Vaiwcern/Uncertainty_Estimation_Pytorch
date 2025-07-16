import matplotlib.pyplot as plt
from metric import corr, rAULC
import os
import numpy as np

def plot_uncertainty_bar_chart(array: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(max(12, len(array) * 0.1), 4))  # tự động giãn theo số lượng phần tử
    plt.bar(range(len(array)), array, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Uncertainty')
    plt.ylim(0, 1)
    plt.grid(axis='y')

    if len(array) <= 30:
        for i, val in enumerate(array):
            plt.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_unc_vs_error(x, y, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=0.4)
    plt.xlabel("Uncertainty (sqrt)")
    plt.ylabel("Prediction Error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 

def plot_corr_rAULC(x, y, title, filename, SAVE_PATH):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=0.4)
    plt.xlabel("Uncertainty")
    plt.ylabel("Error")
    plt.title(f"{title}\nCorr={corr(x, y):.4f} | rAULC={rAULC(x, y):.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, filename))
    plt.close()

def plot_uncertainty_error_bar(uncs, errors, save_path, filename, max_samples=500):
    uncs = np.array(uncs)
    errors = np.array(errors)

    # Sắp xếp theo uncertainty tăng dần
    sorted_indices = np.argsort(uncs)
    uncs_sorted = uncs[sorted_indices]
    errors_sorted = errors[sorted_indices]

    # Giới hạn số lượng sample hiển thị
    if len(uncs_sorted) > max_samples:
        uncs_sorted = uncs_sorted[:max_samples]
        errors_sorted = errors_sorted[:max_samples]

    indices = np.arange(len(uncs_sorted))
    bar_width = 0.8

    plt.figure(figsize=(16, 6))
    plt.bar(indices, uncs_sorted, width=bar_width, color='orange', alpha=0.4, label='Uncertainty')
    plt.bar(indices, errors_sorted, width=bar_width, color='deepskyblue', alpha=0.4, label='Error')

    plt.xlabel("Sample Index (sorted by uncertainty)")
    plt.ylabel("Value")
    plt.title("Uncertainty vs Error (Sorted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def save_as_heatmap(array: np.ndarray, save_path: str, cmap: str = 'plasma'):
    """Save a 2D array as a heatmap image."""
    normalized = (array - array.min()) / (array.max() - array.min() + 1e-8)

    plt.figure(figsize=(4, 4), dpi=100)
    plt.axis('off')
    plt.imshow(normalized, cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
