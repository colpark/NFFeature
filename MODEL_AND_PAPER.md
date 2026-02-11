# Model–Paper Mapping: OmniField (9719) vs. AblationCIFAR10.ipynb

This note maps **“OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning”** (ICLR 2026 under review, `9719_OmniField_Conditioned_Neu (13).pdf`) to the implementation in `AblationCIFAR10.ipynb`.

---

## 1. Paper in one paragraph

OmniField is a **continuity-aware conditioned neural field** for multimodal spatiotemporal data. It avoids gridding and heavy imputation by: (1) an **encoder** that turns irregular, noisy observations into a query-dependent summary; (2) a **processor** that fuses coordinate encodings with that summary into a latent field; (3) **modality decoders** that map the latent to predictions. The main additions over prior CNFs are: **Gaussian Fourier features (GFF)** and **sinusoidal initialization** to reduce low-frequency bias, and **multimodal crosstalk (MCT)** with **iterative cross-modal refinement (ICMR)** to align modalities before decoding. The notebook keeps the same encoder–processor–decoder and GFF/sinusoidal/ICMR ideas but **simplifies to a single spatial modality (CIFAR-10 images)** and a single “context” = full low-res pixel grid.

---

## 2. Paper setup (Section 3) → what the notebook does

| Paper | Notebook (CIFAR ablation) |
|-------|----------------------------|
| **Spatial domain** X, time T | Space only: 2D image grid (x,y). No time. |
| **Modalities** M_all | One “modality”: RGB image. |
| **Context** C = {U_m : m ∈ M_in} | One set: all 32×32 pixels + positions. So “context” = low-res image with coordinates. |
| **Queries** Q = (m, x) at target locations | **Reconstruction:** queries at same 32×32 positions. **Super-resolution:** queries at 128×128 positions. |
| **Tasks** reconstruction / interpolation / forecasting / cross-modal | **Reconstruction** (32→32) and **spatial interpolation / super-resolution** (32→128). No time, no cross-modal. |

So the notebook is the **spatial, single-modality special case**: one context (low-res grid) and query coordinates at 32×32 or 128×128.

---

## 3. Architecture: Encoder–Processor–Decoder

Paper (Section 4):

- **F_θ = {D_ψ,m} ◦ P_θ ◦ E_φ**
- **E_φ**: context C + (x,t) → summary **c(x,t)**
- **P_θ**: coordinates μ(x), λ(t) + c → latent **h(x,t)** (the conditioned neural field)
- **D_ψ,m**: h → **ŷ_m(x,t)**

In the notebook there is no separate “E” that takes a query (x,t). The **context is the full grid** (pixels + positions). So:

| Paper block | Notebook implementation |
|-------------|--------------------------|
| **Encoder E_φ** | The **context** is built once: `prepare_model_input` → `data` = [pixels; pos_embeddings] per token. So “encoding” is “we use the whole 32×32 (pixels + GFF positions) as the set of observations.” No extra query-dependent encoder; the processor attends to this context. |
| **Processor P_θ** | **CascadedPerceiverIO**: `encoder_blocks` (cascaded cross-attn + self-attn) + `self_attn_blocks`. **Input:** `data` (context). **State:** learnable **latents** (sinusoidal-initialized), refined by cross-attention to `data` and self-attention. Output = refined latents = **h**. |
| **Decoder D_ψ,m** | **Decoder** in code: `decoder_cross_attn(queries, context=residual)` → +queries → optional `decoder_ff` → `to_logits` → pixel values. Single “modality” (RGB), so one decoder. |

So: **context** = low-res grid; **processor** = cascaded Perceiver blocks producing a latent field; **decoder** = cross-attention from query coordinates to that latent, then MLP to RGB.

---

## 4. Gaussian Fourier features (GFF) — Section 4.1

Paper:

- Replace fixed sinusoidal features with **Gaussian Fourier features**: sample **B ∈ R^(d×ℓ)** with B_ij ∼ N(0, σ²).
- **μ(x) = concat(cos(2π Bx), sin(2π Bx))** ∈ R^(2ℓ) for richer, less low-frequency-biased representation.

Code:

- **`GaussianFourierFeatures`**: `B` stored as buffer, shape `(in_features, mapping_size)`, scale ~10–15.
- **Forward:** `projections = coords @ B`, then `concat([sin(projections), cos(projections)], dim=-1)`.
- Used for **both**:
  - **Context positions:** in `prepare_model_input`, `pos_embeddings = fourier_encoder(batch_coords)` and concatenated with pixels → **input_dim = CHANNELS + POS_EMBED_DIM**.
  - **Query positions:** `queries = fourier_encoder(batch_coords)` (32×32 or 128×128) → **queries_dim = POS_EMBED_DIM**.

So the same GFF module provides **μ(x)** for context and for queries, matching the paper’s use of GFF for coordinate encoding.

---

## 5. Sinusoidal initialization — Section 4.1

Paper:

- Initialize the **M learnable queries** (latent tokens) with a **multi-scale sinusoidal** pattern (log-spaced bands, unit-norm) for balanced frequency coverage.

Code:

- **CascadedBlock** uses **`get_sinusoidal_embeddings(n_latents, dim)`** for **`self.latents`** (shape `n_latents × dim`), with **`requires_grad=False`**.
- So the **processor’s latent tokens** are **fixed sinusoidal** embeddings, not learnable. The decoder’s “queries” are **not** these tokens; they are **GFF(coords)** (different for each query location).

So “sinusoidal” in the code is the **initialization of the cascaded latents** (the keys/values the decoder will attend to). That matches the paper’s idea of stabilizing and balancing frequency coverage for the latent side; here it’s taken to the extreme (fully fixed).

---

## 6. Multimodal crosstalk (MCT) and ICMR — Section 4.2

Paper:

- **MCT block:** unimodal features **E_m(U_m)** concatenated, then a **multimodal processor** P (e.g. self-attention) and a **global feature z** that aggregates information.
- **ICMR:** for k = 0,…,ℓ−1:  
  **h^(k) = MCT({U_m}, z^(k))**, then **z^(k+1) = (1/n) Σ_i h^(k)_{i,:}**.  
  Start with **z^(0) = 0**. Final field **g = h^(ℓ−1)**.

Code (single-modality CIFAR):

- There are no separate modalities; there is one context `data`.
- **CascadedBlock**:  
  - **Latents** (sinusoidal) attend to **context** (cross-attn), add **residual** from previous block (if any), then self-attn + FF.
- **CascadedPerceiverIO.forward**:
  - For each block: `residual = block(x=residual, context=data, mask=mask, residual=residual)`.
  - Then 4 extra **self_attn_blocks** on `residual`.
  - Decoder: `queries` attend to `residual`, then FF and logits.

So:

- **“Iterative refinement”** = multiple **CascadedBlocks** (and then self-attn blocks) that repeatedly cross-attend to the same context and refine the latent sequence. There is no explicit single-vector **z**; the **residual** (sequence of latents) plays the role of the evolving representation.
- **“MCT”** in the paper is cross-modal exchange; in the notebook it collapses to **cross-attention from latents to the single context** (image + positions). So the code is **one-modality ICMR-like refinement**: multiple layers of “latents ← cross_attn(latents, context) + residual + self_attn + FF”.

---

## 7. Data flow (step by step)

1. **Input**
   - Image batch `(B, 3, 32, 32)`.
   - **Coordinates:** `create_coordinate_grid(32, 32)` → normalized (e.g. [-1,1]) grid `(1024, 2)`.

2. **Context construction** (encoder side)
   - `prepare_model_input(images, coords_32x32, fourier_encoder)`:
     - Pixels: `(B, 1024, 3)`.
     - GFF: `fourier_encoder(batch_coords)` → `(B, 1024, POS_EMBED_DIM)`.
     - **data** = `concat(pixels, pos_embeddings)` → `(B, 1024, INPUT_DIM)` with **INPUT_DIM = 3 + 192** (e.g. for mapping_size=96, 2*96=192).

3. **Query construction**
   - **Reconstruction:** coords 32×32 (optionally with small noise), then `queries = fourier_encoder(batch_coords)` → `(B, 1024, QUERIES_DIM)`.
   - **Super-resolution:** coords 128×128 → `queries` `(B, 16384, QUERIES_DIM)`.

4. **Processor (CascadedPerceiverIO)**
   - **Encoder stack:** 3 × **CascadedBlock**:
     - Latents: fixed sinusoidal `(256, dim)` broadcast to `(B, 256, dim)`.
     - Cross-attn: latents attend to **data** (context).
     - Add residual from previous block (projected if dimension changes).
     - Self-attn + FF → new **residual** for next block.
   - **Self-attn stack:** 4 × (self-attn + FF) on the final residual.
   - Output: **residual** `(B, 256, final_latent_dim)` = the latent field **h**.

5. **Decoder**
   - **Queries** (GFF at query positions) cross-attend to **residual**.
   - Skip: `x = x + queries`.
   - Optional `decoder_ff`, then `to_logits` → **reconstructed/generated pixels** `(B, N_queries, 3)`.

6. **Loss**
   - MSE between decoded pixels and target pixels (same resolution as queries).

So: **context** = one grid of (pixel, position); **queries** = (position only) at 32×32 or 128×128; **processor** = cascaded cross/self-attention; **decoder** = cross-attn from queries to processor output.

---

## 8. Ablations (Table 3 in the paper)

The notebook’s title **“Yes GFF + Yes Sinusoidal + Yes ICMR”** is the **best** configuration in the paper:

- **GFF:** Use **GaussianFourierFeatures** for positions (not fixed log-band Fourier or no GFF).
- **Sinusoidal init:** Use **sinusoidal embeddings** for the cascaded latents (in code: fixed sinusoidal).
- **ICMR:** Use **iterative refinement** (cascaded blocks + self-attn) instead of a single pass or mid-fusion.

The other ablation cells in `_orig` correspond to turning these off (e.g. no GFF, no sinusoidal, no ICMR / mid-fusion).

---

## 9. Unused / legacy in the notebook

- **`input_proj`** and **`projection_matrix`** are defined in **CascadedPerceiverIO** but **never used** in **forward**. So context is fed as-is (pixels + GFF) without an extra linear/GFF projection in the constructor.
- **`FourierFeatures`** (log-spaced bands, MLP) is defined but the active path uses **GaussianFourierFeatures** for the “Yes GFF” ablation.

---

## 10. Summary table

| Paper concept | Where in code |
|---------------|----------------|
| Conditioned neural field (query at arbitrary coordinates) | Decoder: `queries = fourier_encoder(coords)` then cross-attn to processor output. |
| GFF for coordinates | `GaussianFourierFeatures` for context and query positions. |
| Sinusoidal init for latents | `get_sinusoidal_embeddings` in `CascadedBlock.latents` (fixed). |
| Encoder E_φ | Context = full grid (pixels + GFF); no separate E(C; x,t). |
| Processor P_θ | `encoder_blocks` + `self_attn_blocks` (cascaded cross/self-attn). |
| ICMR / iterative refinement | Multiple cascaded blocks + 4 self-attn layers refining latents from one context. |
| Decoder D_ψ,m | `decoder_cross_attn` + skip + `decoder_ff` + `to_logits`. |
| Single modality (CIFAR) | One context tensor, one decoder; no modality masks or MCT over modalities. |

Overall, the notebook is a **spatial, single-modality** instance of OmniField: same encoder–processor–decoder and GFF/sinusoidal/ICMR design, with the full multimodal MCT/ICMR simplified to **cascaded cross-attention from latents to one context grid** and **sinusoidal-initialized latents** instead of learnable queries.
