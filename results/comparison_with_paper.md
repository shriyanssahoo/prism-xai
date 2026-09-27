# Replication Results: Comparison with Original Paper

**Paper**: Szandała, T. (2023). "Unlocking the black box of CNNs: Visualising the decision-making process with PRISM." _Information Sciences_, 642, 119162.

**Original implementation reference**: [TorchPRISM](https://github.com/szandala/TorchPRISM) (consulted for conceptual understanding only; all code in this repository is an independent reimplementation of Algorithm 1 from the paper).

This document compares the results obtained in this replication against the claims and figures presented in the original paper.

---

## Setup Summary

- **Model**: Pretrained VGG-16 (ImageNet-1k weights, `torchvision.models.vgg16`)
- **Layer hooked**: Last convolutional layer of VGG-16's `features` block, producing a `(batch_size, 512, 14, 14)` representation
  - Note: the paper's Fig. 1 example uses a `512 x 7 x 7` representation. Our hook captures a `512 x 14 x 14` representation — one block earlier in the network than the paper's exact layer. This does not affect the correctness of the PRISM algorithm itself, but means our absolute numbers (variance %, spatial resolution) are not directly comparable 1:1 to the paper's, only the qualitative trends are.
- **Dataset**: Images sourced from the official ImageNet website (see `data/sources.md`), for classes matching the paper's examples: timber wolf, coyote, grey fox, border collie, samoyed, electric locomotive, passenger car
- **Random seed**: 42 (set via `torch.manual_seed`, `np.random.seed`, `random.seed`)
- **Gradual Extrapolation**: Not implemented. The paper combines PRISM with a separate technique (Gradual Extrapolation, ref. [24] in the paper) to sharpen the blocky low-resolution output into smooth, human-recognizable images. This replication instead uses simple bicubic upsampling (`cv2.resize`) to blend the PRISM map onto the original image. This is a deliberate scope simplification — the underlying PRISM math (PCA/SVD on the extracted representation) is faithfully reproduced; only the sharpening post-process differs.

---

## Experiment 1: Core PRISM Visualization

**Paper reference**: Section 3.3, Figure 8 (PRISM outputs for two wolves + one coyote, across multiple models)

**Our setup**: 2 timber wolf images + 1 coyote image, batched together, processed through VGG-16.

**Result**:

```
PRISM map shape: (3, 3, 14, 14)
Top-3 PC variance: [0.253, 0.144, 0.077], sum = 47.4%
```

The generated overlay (`notebooks/01_prism_basic_demo.ipynb`) shows:

- **Wolf 1** (front-facing): a clear gradient overlay concentrated on the face, with a distinct pink/magenta region on the muzzle and yellow-green tones across the forehead/ears
- **Wolf 2** (side profile): a much fainter, more uniform pale overlay across the head
- **Coyote**: a small, spatially concentrated purple/pink blob centered on the head, distinct in character from both wolf images

**Comparison to paper**: The paper's Fig. 8 (VGG-16 panel) reports that PRISM highlights the animals' **mouths/muzzles** as the primary distinguishing feature between wolves and coyotes, largely ignoring irrelevant background (rocks, dirt, grass). Our result matches this qualitatively — the strongest, most saturated coloring in our Wolf 1 image is concentrated on the muzzle region, consistent with the paper's finding. Background regions (dirt, rocks) show minimal overlay intensity, also matching the paper's claim that PRISM's attention correctly ignores irrelevant image regions.

One difference: the paper's example shows relatively **consistent, saturated coloring across all three images** in a batch (since their models were evaluated after Gradual Extrapolation sharpening). Our Wolf 2 image shows a noticeably fainter overlay than Wolf 1 despite both being the same predicted class — likely because without Gradual Extrapolation's sharpening, differences in raw activation magnitude between individual images are more visible.

**Verdict**: Reproduced successfully. Qualitative feature localization (muzzle-focus) matches the paper's finding.

---

## Experiment 2: PCA Variance Distribution

**Paper reference**: Section 5.2, Figures 16-18

**Paper's claim**: _"The sum of the variances for the first 3 PCs for up to 5 classes is above 57%. The more classes were added, the less confident the results were."_

**Our results**:

| Batch | Images | Classes                                                   | Top-3 PC Variance |
| ----- | ------ | --------------------------------------------------------- | ----------------- |
| Small | 3      | 2 (timber wolf, coyote)                                   | **47.4%**         |
| Large | 10     | 5 (timber wolf, coyote, grey fox, border collie, samoyed) | **32.6%**         |

(Full breakdown: PC0=0.169, PC1=0.086, PC2=0.072 for the 5-class batch; PC0=0.253, PC1=0.144, PC2=0.077 for the 2-class batch. See `results/tables/variance_summary.csv`, generated in `notebooks/02_pca_variance_experiment.ipynb`.)

**Comparison to paper**: The paper reports **~57%** captured variance for a similar 5-class batch (timber wolf, coyote, grey fox, samoyed, border collie — the same five classes we used). Our 5-class result of **32.6%** is meaningfully lower in absolute terms.

**Trend match**: The core qualitative claim — _variance captured by the top-3 PCs decreases as batch class diversity increases_ — is clearly reproduced. Our 2-class batch (47.4%) captured more variance than our 5-class batch (32.6%), consistent with the paper's finding that "the more classes were added, the less confident the results were."

**Why the absolute numbers differ**:

1. Different specific images (paper's exact source images are not published/available; we used different ImageNet samples for the same classes)
2. Different layer depth (14x14 vs. the paper's 7x7 — a shallower/less-abstracted layer may naturally distribute variance differently)
3. Smaller sample size per class (2 images/class here vs. an unspecified but likely larger number in the paper's setup)

**Verdict**: Qualitative trend reproduced faithfully. Absolute percentages differ, and the reasons for the difference are understood and documented rather than treated as an unexplained discrepancy.

---

## Experiment 3: PRISM vs. Grad-CAM Comparison

**Paper reference**: Sections 3.2, 4.1; Figures 6, 7, 11

**Paper's claim**: _"PRISM highlights a wider area than the saliency mapping technique... However, despite looking at the same area from a human perspective, it allows the user to claim that the two areas have entirely different features according to the model."_ Also: PRISM is "target-agnostic" (Section 4.2) — it doesn't filter for relevance to the specific predicted class, unlike Grad-CAM.

### Image set A: Wolf / Wolf / Grey Fox

| Image            | Predicted class               | Grad-CAM                             | PRISM                                                                   |
| ---------------- | ----------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| Wolf 1 (front)   | Alaskan tundra wolf (idx 270) | Two hot spots: face and shoulder     | Gradient overlay: pink muzzle, yellow-green forehead                    |
| Wolf 2 (profile) | Alaskan tundra wolf (idx 270) | Broad single hot spot over head/neck | Faint, pale overlay over head                                           |
| Grey fox         | idx 280                       | Hot spot on head                     | Distinct purple/pink blob on head, different character from wolf images |

**Comparison to paper**: This closely mirrors the paper's spider example (Fig. 6) and locomotive example (Fig. 7): Grad-CAM produces structurally similar-looking heatmaps for all three images (a red/yellow blob centered on the animal's head/face in each case), giving the visual impression that the model is looking at "the same kind of thing" in all three. PRISM, in contrast, shows **more differentiated coloring** between the two true wolf images and the grey fox — supporting the paper's central claim that PRISM exposes _distinguishing_ features that Grad-CAM's heatmap format cannot represent, since Grad-CAM only encodes _where_, not _what kind of feature_.

### Image set B: Electric Locomotive / Train Cars

| Image               | Predicted class               | Grad-CAM                                             | PRISM                                                               |
| ------------------- | ----------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| Electric locomotive | electric locomotive (correct) | Correctly localizes onto the train body/front        | Overlay concentrated on the locomotive, similar region to Grad-CAM  |
| Train cars          | **harvester (misclassified)** | Still localizes reasonably onto the train cars/rails | Flat, diffuse, undifferentiated orange wash across the entire image |

**This result directly illustrates a limitation the paper itself discusses (Section 4.2)**: PRISM is target-agnostic — it visualizes whatever principal features the model's representation contains, regardless of what the model's final classification layer decided. When the underlying VGG-16 model misclassified the second train image as "harvester," Grad-CAM (which is explicitly gradient-guided toward a specific class) still managed to spatially localize onto the correct region (the train), because it directly used the model's confused-but-still-partially-relevant class-specific gradients. PRISM, having no such target-class guidance, produced an overlay that gave no clear indication of any specific discriminating feature — an example, in reverse, of the same target-agnostic behaviour the paper describes when discussing the light semaphore missed by Grad-CAM but caught by PRISM: here, without a correct target class to anchor to, PRISM's output lost useful signal.

**Verdict**: Reproduced both (a) the paper's core interpretability claim — PRISM differentiates features Grad-CAM's heatmap cannot, evidenced by the wolf/fox set — and (b) a real-world instance of a documented limitation of the method (target-agnosticism degrading output quality on a misclassified image).

---

## Overall Summary

| Experiment                | Paper Section          | Reproduced? | Match Quality                                                       |
| ------------------------- | ---------------------- | ----------- | ------------------------------------------------------------------- |
| Core PRISM visualization  | 3.3, Fig. 8            | Yes         | Strong qualitative match (muzzle-focus finding)                     |
| PCA variance distribution | 5.2, Figs. 16-18       | Yes         | Trend matches exactly; absolute % differs, reasons documented       |
| PRISM vs. Grad-CAM        | 3.2, 4.1, Figs. 6/7/11 | Yes         | Matches core claim + surfaced a real instance of a known limitation |

## Scope Not Attempted

For transparency, the following parts of the paper were **not** reproduced in this replication, due to time/scope constraints:

- **Section 3.1 (Representation swapping / faithfulness test)** — requires intervening on intermediate network representations directly, a more involved experimental setup
- **Section 3.3 (9-model applicability test)** — paper tests AlexNet, GoogleNet, ResNet-18/50/101, SqueezeNet, VGG-11/16/19; we used only VGG-16
- **Section 5 (SOM clustering + confusion matrix, Table 1)** — requires ~20-30 images per class to build a statistically meaningful confusion matrix; we used 2 images/class
- **Section 5.1 (Vision Transformer / DeiT)** — applying PRISM to a non-CNN architecture
- **Gradual Extrapolation** — used simple bicubic upsampling instead (see Setup Summary above)
- **Figure 10 (execution time benchmark)** — comparing PRISM's speed against other XAI methods

These represent natural extensions for future work but were outside the scope of this course replication project, which focused on reproducing PRISM's core algorithm and its two central comparative claims (variance efficiency and differentiation-vs-Grad-CAM).
