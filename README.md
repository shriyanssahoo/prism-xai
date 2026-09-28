# PRISM Replication: Visualising the Decision-Making Process of CNNs

Replication of the key experiments from:

> Szandała, T. (2023). _Unlocking the black box of CNNs: Visualising the decision-making process with PRISM._ Information Sciences, 642, 119162. https://doi.org/10.1016/j.ins.2023.119162

This project is Phase 1 (Replication) of an Explainable AI (XAI) course project. It re-implements the **PRISM (Principal Image Sections Mapping)** method from scratch, applies it to a pretrained VGG-16, and reproduces three of the paper's main results.

---

## 1. What is PRISM?

PRISM explains what a CNN "sees" by applying Principal Component Analysis (PCA) to a layer's feature maps across a whole **batch** of images at once:

1. Extract a 4D representation `(batch, channels, H, W)` from a chosen CNN layer
2. Reshape to a 2D matrix `channels x (batch * H * W)` and centre it
3. Run SVD (PCA) and keep the top 3 principal components
4. Reshape back and use the 3 components as the R, G, B channels of an image mask

Because PCA is computed jointly over the batch, the same colour in different images means the model is treating those regions as the same feature. This exposes _distinguishing_ features between images, which single-image saliency maps (e.g. Grad-CAM) cannot show.

---

## 2. Experiments Reproduced

| #   | Experiment                                           | Paper reference                | Notebook                                     |
| --- | ---------------------------------------------------- | ------------------------------ | -------------------------------------------- |
| 1   | Core PRISM visualisation (2 wolves + 1 coyote)       | Sec. 3.3, Fig. 8               | `notebooks/01_prism_basic_demo.ipynb`        |
| 2   | PCA variance distribution (2-class vs 5-class batch) | Sec. 5.2, Figs. 16-18          | `notebooks/02_pca_variance_experiment.ipynb` |
| 3   | PRISM vs Grad-CAM comparison                         | Sec. 3.2 / 4.1, Figs. 6, 7, 11 | `notebooks/03_prism_vs_gradcam.ipynb`        |

### Results summary

| Experiment              | Our result                                                                                                                                                                                                                      | Paper                                                                          | Match                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| 1. Core PRISM           | Strongest colour response on the wolf's muzzle; background largely ignored                                                                                                                                                      | Muzzle identified as the distinguishing feature                                | Qualitative match                                    |
| 2. Variance (top-3 PCs) | 2-class (3 images): **47.4%**; 5-class (10 images): **32.6%**                                                                                                                                                                   | "Above 57%" for up to 5 classes; more classes means lower variance             | Trend matches; absolute values lower                 |
| 3. PRISM vs Grad-CAM    | Grad-CAM gives similar head blobs for all animals; PRISM colours differ between wolves and fox. On a misclassified train image ("harvester"), PRISM gave a flat, uninformative overlay while Grad-CAM still localised the train | PRISM highlights differentiating features; PRISM is target-agnostic (Sec. 4.2) | Core claim and a documented limitation both observed |

A detailed side-by-side discussion is in [`results/comparison_with_paper.md`](results/comparison_with_paper.md). All output figures are embedded in the notebooks and render directly on GitHub.

---

## 3. Deviations from the Paper (please read)

- **Gradual Extrapolation not implemented.** The paper pairs PRISM with Gradual Extrapolation (Szandała, 2021) to sharpen outputs. We use bicubic upsampling (`cv2.resize`) instead, so our overlays are smoother/blurrier than the paper's Fig. 8. The PRISM algorithm itself (PCA on the representation) is unchanged.
- **Layer / resolution.** We hook the last `Conv2d` of VGG-16 (`model.features[28]`), giving a `512 x 14 x 14` representation (before the final max-pool). The paper's example uses `512 x 7 x 7`. Absolute variance values are therefore not directly comparable; only trends are.
- **Different images.** The paper's exact images are not published. We used ImageNet samples of the same classes, with only 2 images per class in the 5-class batch.
- **Single model.** Only VGG-16 was used (paper tests 9 architectures).

### Not attempted

Representation swapping (Sec. 3.1), 9-model applicability test (Sec. 3.3), SOM clustering and confusion matrix (Sec. 5, Table 1), Vision Transformer / DeiT (Sec. 5.1), and the execution-time benchmark (Fig. 10).

---

## 4. Repository Structure

```
.
├── README.md
├── AI_USAGE.md                 # AI tool usage disclosure
├── requirements.txt
├── data/
│   ├── sources.md              # dataset provenance (ImageNet wnids)
│   └── raw/
│       ├── prepare_dataset.py  # extracts N images per class from ImageNet tars
│       └── <class_name>/       # extracted images (timber_wolf, coyote, ...)
├── src/
│   ├── model_utils.py          # model loading, preprocessing, forward hook
│   ├── prism.py                # PRISM algorithm (Algorithm 1 of the paper)
│   ├── visualize.py            # upsampling / overlay / comparison plots
│   ├── variance_analysis.py    # PCA variance distribution plots
│   └── gradcam_baseline.py     # Grad-CAM baseline wrapper
├── notebooks/
│   ├── 01_prism_basic_demo.ipynb
│   ├── 02_pca_variance_experiment.ipynb
│   └── 03_prism_vs_gradcam.ipynb
└── results/
    └── comparison_with_paper.md
```

---

## 5. Setup and Reproduction

### Environment

Developed and tested on **Google Colab (T4 GPU)**. Inference only, so it also runs on CPU. **Python 3.10+**.

### Installation

```bash
git clone https://github.com/<YOUR_USERNAME>/prism-xai.git
cd prism-xai
pip install -r requirements.txt
```

> Note: the Grad-CAM library is installed with `pip install grad-cam` (PyPI name) and imported as `pytorch_grad_cam`.

### Dataset

Images come from **ImageNet** (https://image-net.org), which requires a registered account. Per-class archives are downloaded from:

```
https://image-net.org/data/winter21_whole/{wnid}.tar
```

The wnids used are listed in [`data/sources.md`](data/sources.md). To extract 10 images per class:

```bash
cd data/raw          # folder containing the downloaded {wnid}.tar files
python prepare_dataset.py --n_images 10
```

This creates `data/raw/<class_name>/*.JPEG`, which is what the notebooks expect. The extracted images are already included in this repository.

### Running the experiments

Open and run the notebooks in order (`Restart & Run All`):

1. `notebooks/01_prism_basic_demo.ipynb`
2. `notebooks/02_pca_variance_experiment.ipynb`
3. `notebooks/03_prism_vs_gradcam.ipynb`

Run them from the repository root (in Colab: `%cd /content/prism-xai`). Notebooks 2 and 3 save figures/tables under `results/figures/` and `results/tables/`, creating the folders if needed.

### Random seed

**Seed = 42** (`torch.manual_seed`, `np.random.seed`, `random.seed`). PRISM (SVD) and the pretrained-model inference are deterministic; the seed is set for completeness. Image selection is deterministic per the file order returned by `glob`.

---

## 6. Implementation Notes

- `src/prism.py` is an **independent re-implementation of Algorithm 1** from the paper (reshape, centre, SVD, keep 3 PCs, reshape back, per-component min-max normalisation for display). It also returns the full variance spectrum used for Experiment 2.
- `src/model_utils.py` uses a PyTorch forward hook to capture intermediate activations.
- `src/gradcam_baseline.py` wraps the `pytorch-grad-cam` library rather than re-implementing Grad-CAM.

---

## 7. References and Credits

- **Paper:** T. Szandała, "Unlocking the black box of CNNs: Visualising the decision-making process with PRISM," _Information Sciences_ 642 (2023) 119162.
- **Original PRISM conference paper:** T. Szandała, H. Maciejewski, "PRISM: Principal Image Sections Mapping," ICCS 2022, Springer, pp. 749-760.
- **Author's reference implementation:** [TorchPRISM](https://github.com/szandala/TorchPRISM). It was consulted for conceptual understanding only; no code was copied. All code here was written independently from the paper's pseudocode.
- **Gradual Extrapolation:** T. Szandała, "Enhancing deep neural network saliency visualizations with gradual extrapolation," IEEE Access 9 (2021) 95155-95161 (not implemented here).
- **Grad-CAM:** R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," ICCV 2017.
- **Grad-CAM library:** J. Gildenblat and contributors, [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
- **VGG-16:** K. Simonyan, A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv:1409.1556, 2014 (torchvision pretrained weights).
- **ImageNet:** J. Deng et al., "ImageNet: A large-scale hierarchical image database," CVPR 2009. Images used for non-commercial educational purposes.
- **ImageNet label list** (for readable class names): [imagenet-simple-labels](https://github.com/anishathalye/imagenet-simple-labels).

---

## 8. AI Usage

AI assistance (Claude) was used for planning, code drafting, debugging and documentation. Full details (tools, prompts, and how outputs were modified) are in [`AI_USAGE.md`](AI_USAGE.md).
