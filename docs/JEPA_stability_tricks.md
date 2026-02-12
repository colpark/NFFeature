# JEPA / I-JEPA: Stability Tricks from the Literature

Summary of practices and tricks to keep JEPA training stable and avoid representation collapse, drawn from the I-JEPA paper, V-JEPA, C-JEPA, SALT, and related work.

---

## 1. Collapse and why it happens

- **JEPA** trains by predicting target embeddings from context; the **target** is given by an EMA (target) encoder. There are **no contrastive negatives**.
- Risk: the predictor can drive the online encoder toward outputs that are easy to predict (e.g. constant or low-rank), giving low loss but **collapsed** representations.
- EMA alone is often **not enough** to prevent full collapse; the target can still track the student into a collapsed regime.

### 1.1 How “training collapse” happens mechanistically

I-JEPA’s main collapse avoidance is **asymmetry**:

- **Context (“student”) encoder**: trained by **gradient** (loss → z_pred → predictor → z_c → online encoder).
- **Target (“teacher”) encoder**: updated only by **EMA** of the student, and **stop-gradient** on the target branch (target is a fixed label for this step).

If you break that asymmetry, the system can **chase itself** into the constant-vector shortcut:

1. **No stop-grad on target**: If the target branch is inside the autograd graph, gradients flow into the target encoder. Then both encoders get the same pressure (“match each other”). The easiest way to match is for both to output the same constant; gradients can then push both toward that constant. So **always** compute z_t_target under `torch.no_grad()` (or `.detach()`), and **never** put target encoder parameters in the optimizer.

2. **Updating both encoders by backprop**: If both online and “target” are in the optimizer and receive gradients from the same loss, they co-adapt: the loss is minimized when z_pred ≈ z_t, which is trivially achieved by making both encoders output a constant. So the target must be **EMA-only**, not updated by gradient.

3. **EMA momentum misconfigured**: If the teacher (EMA) changes **too fast** (e.g. momentum too low, so teacher ≈ student every step), the target tracks the student almost exactly. Then the predictor’s job is easy: “predict what the student will output,” and again the shortcut is for the student to output a constant so the predictor can match it. So use a **slow** EMA (e.g. momentum 0.996–0.999) so the target is a stable, lagging copy of the student.

**Summary**: Keep (a) **stop-grad** on the target branch, (b) **target encoder not in the optimizer** (updated only by `copy_ema`), and (c) **slow EMA** so the teacher doesn’t chase the student into collapse.

---

## 2. Anti-collapse mechanisms

### 2.1 Asymmetric design (EMA target)

- **I-JEPA (Assran et al., CVPR 2023)**: Target encoder is an **exponential moving average** of the context encoder; no gradients through the target. This asymmetry helps stability.
- **Momentum schedule**: Some setups use a **ramped** EMA (e.g. momentum 0.996 → 1.0 over training) so the target changes more early and stabilizes later.

### 2.2 Prediction in latent space (not pixels)

- Predicting in **representation space** (φ or patch tokens) instead of pixels pushes the model to learn **semantic** structure and reduces the incentive to collapse to trivial pixel-level predictors.
- I-JEPA uses **L2** (not L1) between predicted and target patch representations in the paper; implementations sometimes use smooth L1. Both are regression in embedding space.

### 2.3 Mask design (context vs target)

- **Target blocks**: I-JEPA uses **multiple** target blocks (e.g. 4), each with **large enough scale** (e.g. 15–20% of image) and **aspect ratio** (e.g. 0.75–1.5) so the task is **semantic** rather than local texture.
- **Context block**: One (or more) context block(s) with **high coverage** (e.g. 85–100% of image) and **unit aspect** so the context is informative and spatially distributed.
- **Non-overlap**: Context is often defined in the **complement** of target blocks so the prediction task is non-trivial (predict unseen regions from seen ones).
- **Mutual information**: Poor context/target design (e.g. low mutual information between context and target) can encourage collapse; masks should support a meaningful prediction task.

### 2.4 Explicit variance/covariance regularization (VICReg-style)

- **C-JEPA (NeurIPS 2024)**: Combines I-JEPA with **VICReg** (variance + covariance terms) on the **online** embeddings to prevent collapse and encourage diverse, decorrelated dimensions.
- **Variance**: Encourage per-dimension std above a threshold (e.g. 1) over the batch.
- **Covariance**: Penalize off-diagonal covariance so dimensions do not replicate the same information.
- **Weight**: Balance VICReg with the JEPA loss (e.g. small λ so JEPA still drives learning; too large and VICReg can dominate and slow useful learning).

### 2.5 Frozen teacher (SALT)

- **SALT** and related work: Train a **frozen** target encoder (e.g. with a reconstruction objective) first, then train the student to predict the frozen target. This removes co-adaptation and can improve stability and transparency.
- Trade-off: no end-to-end EMA update; target distribution is fixed after stage 1.

### 2.6 Predictor capacity and depth

- I-JEPA uses a **narrow, deep** predictor (e.g. 12 layers, 384 dim) so the predictor can express complex mappings from context to target positions without the encoder collapsing to satisfy a too-simple predictor.
- Position conditioning (mask/position tokens) is important so the predictor knows **where** it is predicting.

### 2.7 Heavy masking (V-JEPA)

- In **V-JEPA**, very **heavy masking** (e.g. up to 90% of tokens) is used so the model must rely on semantic structure rather than local cues, which can reduce collapse and improve semantics.

---

## 3. Practical checklist for our setup

**Target encoder gradient check (our notebook):**

- **Stop-grad:** The entire target branch is computed inside `with torch.no_grad():` (full_input/residual_ema → model_ema, fourier_ema → phi_t_raw → semantic_head_ema → z_t_target). So no graph is built for the target; gradients never flow into `model_ema`, `fourier_ema`, or `semantic_head_ema`.
- **Target not in optimizer:** `params = [model, fourier_encoder, semantic_head, predictor, mask_rgb]` only; `model_ema`, `fourier_ema`, `semantic_head_ema` are **not** in the optimizer. They are updated only by `copy_ema(..., current_momentum)` after each `optimizer.step()`.
- **EMA update:** `copy_ema` uses `p_t.data.mul_(momentum).add_(p_s.data, alpha=1-momentum)` (in-place on `.data` only), so no gradients involved.

| Trick | Our JEPA notebook | Recommendation |
|-------|--------------------|----------------|
| EMA target + stop-grad | ✅ `no_grad` on target branch; EMA not in optimizer | Keep; do not remove no_grad or add EMA to optimizer |
| EMA momentum | ✅ `copy_ema`, ramp 0.996→0.999 | Keep slow teacher |
| Predict in φ (latent) | ✅ JEPA loss on z_pred vs z_t_target | Keep |
| Mask: large target blocks | ✅ `PRED_MASK_SCALE=(0.15, 0.25)`, block masking | Keep; avoid very small target regions |
| Mask: informative context | ✅ `ENC_MASK_SCALE=(0.85, 1.0)`, complement of target | Keep |
| VICReg | ✅ `vicreg_loss` on z_c | Keep; try lowering `LAMBDA_VICREG` (e.g. 0.01–0.02) if JEPA collapses too fast |
| Loss type | Cosine | I-JEPA uses L2; try smooth L1 if you want closer match |
| Monitor collapse | ✅ `phi_std_mean`, `phi_std_min`, `knn_acc` every epoch | Use to early-stop or reduce VICReg weight if knn_acc drops while jepa is very low |
| RGB auxiliary | ✅ Small λ reconstruction | Helps anchor representations; keep |
| Frozen teacher (SALT) | ❌ | Optional: pretrain encoder with reconstruction, then freeze and train predictor only |

---

## 4. References (key papers / code)

- **I-JEPA**: Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” CVPR 2023. (arXiv:2301.08243) — L2 loss, 4 target blocks 15–20%, context 85–100%, EMA target.
- **V-JEPA**: Video JEPA; EMA, L1 loss, heavy masking (e.g. 90%).
- **C-JEPA**: Mo & Tong, “Connecting JEPA with Contrastive SSL” (VICReg + I-JEPA), NeurIPS 2024 — explicit variance/covariance to prevent collapse.
- **SALT**: “Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers” — two-stage, frozen target encoder.
- **facebookresearch/ijepa** (and **jepa**): Official I-JEPA and V-JEPA code (mask collator, EMA, predictor).

---

## 5. Summary

Stability comes from: **(1)** asymmetric EMA target, **(2)** prediction in latent space with a non-trivial mask design (large semantic target blocks, informative context), **(3)** optional explicit anti-collapse regularization (e.g. VICReg), **(4)** monitoring representation quality (e.g. k-NN, per-dim std) and adjusting loss weights or stopping when collapse is detected.
