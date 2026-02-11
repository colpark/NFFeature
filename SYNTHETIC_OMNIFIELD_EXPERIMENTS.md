# Synthetic Experiments: Proving OmniField Properties

This document designs **synthetic-data experiments** to test four claims about the OmniField (coordinate-conditioned neural field) framework. Each experiment uses **known ground truth** so we can measure success directly.

---

## 1. Embedding geometry directly into representation space

**Claim:** The representation encodes spatial/geometric structure; similarity in representation space reflects geometric relationship (e.g., same physical point under a known warp), not just appearance.

### Design

- **Synthetic data**
  - **View A:** 2D image with known structure (e.g., checkerboard, radial gradient, or simple shapes on a grid).
  - **View B:** Same content under a **known warp** T (affine or thin-plate). Optionally add small noise.
  - Pairs (A, B) with **ground-truth correspondence**: for each point x in A, true match in B is y = T(x).

- **OmniField setup**
  - Encode A and B separately (same model): `residual_a = get_residual(model, input_a)`, `residual_b = get_residual(model, input_b)`.
  - **Probes:** For anchors {x_i} in A, get φ_A(x_i) = project(decoder(q(x_i), residual_a)). In B, get φ_B(y) on a grid (or dense set) of candidates y.
  - No class labels; only geometry (coordinates + warp T).

### Metrics

- **Correspondence accuracy (primary):** For each anchor x_i, rank candidates y by similarity(φ_A(x_i), φ_B(y)). Success = **rank of true match T(x_i)** (lower is better) or **Recall@k** (true match in top-k).
- **Ablation:** Compare OmniField (coordinate-conditioned queries) vs. a **baseline with no coordinates** (e.g., global pooled feature per image). Only the geometry-aware model should achieve high recall.

### How to prove

- **Positive:** OmniField Recall@1 or mean rank significantly better than the no-coordinate baseline on the same synthetic pairs.
- **Control:** Shuffle the warp (wrong T) and show that accuracy drops to chance, confirming that the model is using geometry, not just memorizing image identity.

---

## 2. Supporting multi-scale semantics naturally

**Claim:** The same representation supports semantics at multiple scales (e.g., local texture vs. global shape) without an explicit multi-scale architecture.

### Design

- **Synthetic data**
  - Images with **hierarchical structure**: e.g., large shape (scale 1) containing smaller shapes (scale 2), or multi-scale patterns (fractal-like or nested circles/squares).
  - **Labels:** At least two scales, e.g. “region at scale s has label L” (inside/outside big shape; inside/outside small shape).

- **OmniField setup**
  - Train (or use pretrained) model on reconstruction. No scale-specific heads.
  - **Probing at multiple scales:** (1) Query at **coarse grid** (e.g., 8×8) and pool → “global” descriptor. (2) Query at **fine grid** (e.g., 32×32) in a local patch → “local” descriptor. (3) Optionally use different **Fourier scales** (different GFF scale) for queries to emphasize low vs. high frequency.

### Metrics

- **Consistency across scales:** For the same spatial region, compare predictions (e.g., inside/outside a shape) when probed at coarse vs. fine resolution. Agreement rate.
- **Task:** Simple classifier on pooled φ (e.g., “which of two shapes contains this point?”) at scale 1 and scale 2. Accuracy at each scale without scale-specific training.

### How to prove

- **Positive:** Single model achieves good accuracy at both coarse and fine scale; performance does not collapse when query resolution or GFF scale changes.
- **Ablation:** Compare to a **fixed-resolution baseline** (e.g., CNN that only sees one resolution); show that the baseline degrades when evaluated at another resolution while OmniField stays stable.

---

## 3. Decoupling representation from discretization

**Claim:** The representation is a **continuous function** of coordinates; quality does not depend on the specific grid used for querying.

### Design

- **Synthetic data**
  - **Continuous ground truth:** Define a scalar (or vector) field on [-1,1]², e.g. f(x,y) = distance to a curve, or a smooth analytic function. Render or sample it on a **training grid** (e.g., 32×32).
  - Train OmniField to reconstruct this field from the discrete observations (context = grid + values).

- **Evaluation at multiple resolutions**
  - **Same resolution:** Query at 32×32 → MSE vs. ground truth.
  - **Different resolution:** Query at 64×64, 128×128, or **random sub-pixel coordinates**; compute MSE vs. the **continuous** ground truth (or high-res reference).
  - **Discrete baseline:** A model that only outputs a fixed grid (e.g., CNN encoder–decoder at 32×32). Evaluate at 64×64 by interpolation; compare MSE.

### Metrics

- **MSE at training resolution** (sanity check).
- **MSE at unseen resolutions** (64×64, 128×128, random points).
- **Relative performance:** OmniField vs. discrete baseline when both are evaluated off their native grid.

### How to prove

- **Positive:** OmniField maintains low MSE at all query resolutions; discrete baseline degrades when evaluated at a different grid or when we query at random sub-pixel locations (discrete model has no true value there without interpolation).
- **Smoothness:** Optionally measure smoothness of the decoded field (e.g., Lipschitz constant or variance of gradients) to show the representation is continuous rather than grid-artifact.

---

## 4. Enabling continuous correspondence refinement

**Claim:** The model supports **refining** a correspondence in a continuous way (e.g., gradient ascent on similarity in coordinate space) to get sub-pixel accuracy.

### Design

- **Synthetic data**
  - Same as in (1): pairs (A, B) with known warp T. Optionally add noise or small deformations.

- **OmniField setup**
  - For anchor x in A, we have φ_A(x). In B, define a **similarity map** S(y) = similarity(φ_A(x), φ_B(y)) for y in a grid. Ground truth: y* = T(x).
  - **Refinement:** Start from an **initial guess** y0 (e.g., from argmax on a coarse grid, or y0 = y* + noise). Treat S as a function of **continuous** y (by decoding φ_B at arbitrary y via the continuous query interface). Run **gradient ascent** on y to maximize S(y). Final y_final ≈ y*?

### Metrics

- **Initial error:** ‖y0 − y*‖ (e.g., before refinement).
- **Final error:** ‖y_final − y*‖ after N steps of refinement.
- **Improvement:** Ratio or difference of errors; fraction of points that reach sub-pixel error after refinement.

### How to prove

- **Positive:** Refinement consistently reduces error; many points reach sub-pixel accuracy. Compare to **no refinement** (argmax on a fixed grid only) and to **discrete refinement** (e.g., local search on a fixed grid); show that continuous refinement wins.
- **Requires differentiability:** φ_B(y) must be differentiable w.r.t. y (decoder receives GFF(y), and GFF is differentiable). So we need to implement `y.requires_grad_(True)` and backprop through the decoder to y.

---

## Implementation outline (OmniField)

- **Shared:** Use existing `CascadedPerceiverIO`, `GaussianFourierFeatures`, `prepare_model_input`, `get_residual`, and decoder (or `get_rgb_and_phi_raw` with projection head for φ).
- **Synthetic data loaders:** (1) Warped pairs (A, B, T). (2) Continuous field f(x,y) + discrete sampling. (3) Multi-scale labeled regions.
- **Experiments 1 & 4:** Same data; Exp 1 = evaluation of correspondence (rank/Recall@k); Exp 4 = add refinement loop (differentiate through decoder w.r.t. y).
- **Experiment 2:** Multi-scale probe (coarse vs. fine grids, optional GFF scale); simple classifier or consistency metric.
- **Experiment 3:** Train on a continuous field (or image) at one resolution; evaluate at multiple resolutions and at random coordinates; compare to a discrete baseline.

---

## Suggested order

1. **Experiment 3 (Decoupling from discretization)** — Easiest to implement: train on 32×32, evaluate at 64×64 and random points; no second view.
2. **Experiment 1 (Geometry in representation)** — Reuse Soft InfoNCE-style setup; synthetic pairs with known T; measure rank/Recall@k.
3. **Experiment 4 (Continuous refinement)** — Build on Exp 1; add differentiable y and gradient-ascent refinement; report error before/after.
4. **Experiment 2 (Multi-scale semantics)** — Requires multi-scale labeled synthetic data; then probe at two scales and measure consistency/accuracy.

This order gives quick wins (3, 1) and then extends to refinement (4) and multi-scale (2).
