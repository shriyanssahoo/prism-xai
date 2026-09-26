"""
Core PRISM implementation (Algorithm 1 from Szandała, 2023,
"Unlocking the black box of CNNs: Visualising the decision-making
process with PRISM", Information Sciences 642).

This is an independent reimplementation of the published pseudocode,
not a copy of the author's TorchPRISM source. Cite original paper
and repo in README.
"""
import torch
import numpy as np


def compute_prism(activations: torch.Tensor, n_components: int = 3):
    """
    Args:
        activations: 4D tensor (batch_size, channels, H, W) -- the
            representation extracted from a chosen CNN layer for a
            BATCH of images processed together.
        n_components: number of principal components to keep (3 for RGB).

    Returns:
        prism_maps: np.ndarray (batch_size, n_components, H, W), each
            channel normalized to [0, 1] for display.
        variances: np.ndarray (n_components,) fraction of variance
            explained by each of the kept components.
        all_variances: np.ndarray, full variance spectrum (for the
            variance-distribution experiment, Sec 5.2 of the paper).
    """
    batch_size, channels, H, W = activations.shape

    # Step 1: reshape to 2D matrix -> channels x (batch*H*W)
    # each channel becomes one "feature", each spatial-batch position one "sample"
    x = activations.permute(1, 0, 2, 3).reshape(channels, -1)  # (channels, batch*H*W)
    x = x.cpu().numpy().astype(np.float64)

    # Step 2: center the data (mean over the channel dimension per sample)
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean

    # Step 3: SVD (equivalent to PCA on centered data)
    # centered: (channels, N) -> U: (channels, channels), S: (channels,), Vt: (channels, N)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # full variance spectrum, for Fig 16/17/18-style plots
    all_variances = (S ** 2) / np.sum(S ** 2)

    # Step 4: take top-3 PCs. Projected data = U^T @ centered gives PC scores
    # per original "channel" -> we actually want the PCs as new "channel" maps,
    # i.e. project the ORIGINAL data onto the top components:
    pcs = U[:, :n_components].T @ centered  # (n_components, N)

    # Step 5: reshape back to (batch_size, n_components, H, W)
    pcs = pcs.reshape(n_components, batch_size, H, W).transpose(1, 0, 2, 3)

    # Step 6: normalize each component to [0,1] per-batch for RGB display
    prism_maps = np.zeros_like(pcs)
    for c in range(n_components):
        channel = pcs[:, c, :, :]
        cmin, cmax = channel.min(), channel.max()
        prism_maps[:, c, :, :] = (channel - cmin) / (cmax - cmin + 1e-8)

    variances = all_variances[:n_components]
    return prism_maps, variances, all_variances