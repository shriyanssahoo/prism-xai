%%writefile src/variance_analysis.py
"""
src/variance_analysis.py

Reproduces the PCA variance distribution experiment from the paper
(Section 5.2, Figures 16-18). Shows how much of the total variance
in a CNN's feature representations is captured by the top-3 PCs
that PRISM uses for its RGB visualization.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_variance_distribution(all_variances, top_n_pcs=3, title="PCA Variance Distribution",
                                 save_path=None, max_components_to_show=50):
    """
    all_variances: 1D np.ndarray of variance fractions per component,
        sorted descending (this is what compute_prism() returns as
        its third output).
    top_n_pcs: how many top components PRISM actually keeps (default 3)
    max_components_to_show: truncate the x-axis for readability,
        since there can be hundreds of components (e.g. 512 for VGG-16)
    """
    variances_to_plot = all_variances[:max_components_to_show]
    top_sum = all_variances[:top_n_pcs].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if i < top_n_pcs else "#1f77b4" for i in range(len(variances_to_plot))]
    ax.bar(range(len(variances_to_plot)), variances_to_plot, color=colors)

    ax.set_xlabel("Principal Component index")
    ax.set_ylabel("Fraction of variance explained")
    ax.set_title(title)

    textstr = "\n".join([f"PC {i}: {all_variances[i]:.3f}" for i in range(top_n_pcs)])
    textstr += f"\n\nTop-{top_n_pcs} sum: {top_sum:.3f} ({top_sum*100:.1f}%)"
    ax.text(0.65, 0.75, textstr, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9, verticalalignment="top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()

    return top_sum


def compare_batch_sizes(variance_results: dict, top_n_pcs=3, save_path=None):
    """
    variance_results: dict mapping a label (e.g. "2 classes", "5 classes")
        to that batch's all_variances array. Produces a side-by-side
        comparison, similar to the paper's Fig 16 (two-panel comparison).
    """
    n = len(variance_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    summary = {}
    for ax, (label, variances) in zip(axes, variance_results.items()):
        max_show = min(50, len(variances))
        v = variances[:max_show]
        top_sum = variances[:top_n_pcs].sum()
        summary[label] = top_sum

        colors = ["#d62728" if i < top_n_pcs else "#1f77b4" for i in range(len(v))]
        ax.bar(range(len(v)), v, color=colors)
        ax.set_title(f"{label}\nTop-{top_n_pcs} variance: {top_sum*100:.1f}%")
        ax.set_xlabel("PC index")
        ax.set_ylabel("Variance fraction")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()

    return summary