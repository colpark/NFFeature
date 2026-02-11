# Unsupervised semantics: brainstorm (no labels)

Goal: make the field / TDSM more **semantic** (object vs background, structure, parts) without using class labels. NCE already adds geometry and some object/background differentiation; here we consider ways to go further.

---

## 1. Sparse-based reconstruction

**Idea**: Sparsity often forces **factorization** and **interpretable factors** (sparse coding, ICA). If the model must explain the image with fewer "active" elements, it may learn more semantic building blocks.

- **Sparse context**: Train with **sparse observations**—e.g. only 10–20% of pixels (or a few random patches) as context—and reconstruct the full image. With little context, the model cannot copy texture; it must rely on **structure and prior**. That prior tends to be more semantic (objects, layout).  
  - Implementation: mask a random subset of grid positions when building `data`; still predict at all positions. Or use only a random subset of coordinates as context tokens.

- **Sparse latent usage**: Encourage each query to use only a **subset** of the 256 latent tokens (e.g. softmax attention with **low temperature** → peaked/sparse attention, or L1 on attention weights). Different spatial locations might then "select" different tokens → tokens can **specialize** (e.g. some for object, some for background, some for parts).

- **Sparse decoding**: Reconstruct using only **top-k** tokens per query (e.g. top-k by attention weight, rest masked). Backprop through the selection (e.g. straight-through or soft top-k). Sparsity pressure can yield more semantic tokens.

- **Sparse probes**: Query the field at **sparse** coordinates (e.g. 256 random points) instead of a dense grid; reconstruct only at those points. Or use **importance-weighted** sampling (e.g. more samples where error is high). The model may allocate capacity to "important" (often semantic) regions.

**Verdict**: Sparse **context** is one of the most promising: it directly forces the model to rely on global/structure prior rather than local texture. Sparse **latent** usage can encourage token specialization and might separate object vs background tokens without labels.

---

## 2. Multi-view / augmentation structure (still unsupervised)

- **Multi-crop consistency**: Two **random crops** from the same image; encourage φ (or reconstruction) to be consistent where they **overlap**. Overlap often contains the object; the model learns to keep that region consistent → object-aware. (SimCLR / MoCo style, but you can apply it to your field’s φ or to reconstruction.)

- **Scale consistency**: Reconstruct (or match φ) at **multiple scales** (e.g. 32×32 and 16×16). Structure persists across scale; texture does not. Matching across scales can bias the representation toward structure.

- **Augmentation-aware NCE**: You already use affine views with known T. You could add **stronger augmentations** (e.g. color jitter, cutout) on one view only; the other view is geometric only. The model must match using **geometry and structure**, not color/texture → more semantic alignment.

---

## 3. Reconstruction targets that favor structure

- **Edge / structure auxiliary loss**: Derive **edges** from the image (e.g. Sobel, or a fixed edge detector). Auxiliary loss: predict edges from φ or from a decoded "structure" map. No labels—edges are unsupervised. Pushes the representation to encode **boundaries** (object vs background).

- **Blurred / low-res consistency**: Reconstruct a **blurred** or **downsampled** version of the image (e.g. 8×8 then upsample to 32×32). Blur removes texture and keeps rough structure. Matching reconstruction to blurred target can emphasize structure over texture.

- **Two-stream**: One stream reconstructs **pixels**; another stream reconstructs a **handcrafted structure** (e.g. edges, or PCA of patches). Shared encoder; the encoder is encouraged to carry structure.

---

## 4. Clustering / self-supervision (no human labels)

- **Deep clustering**: Cluster φ (or pooled TDSM) with **k-means** (or similar); use **cluster assignments as pseudo-labels** and add a cross-entropy or contrastive loss toward those assignments. Iterate: cluster → train → re-cluster. If the representation is already somewhat semantic (e.g. after NCE), clusters can become more semantic over time (chicken-and-egg; NCE can bootstrap).

- **Pseudo-label consistency**: An **EMA** (exponential moving average) of the model produces soft or hard pseudo-labels (e.g. nearest-neighbor in a feature bank, or cluster id). Current model is trained to match. Unsupervised, but can sharpen semantics.

---

## 5. Information bottleneck / compactness

- **Bottleneck**: Constrain the **capacity** of the latent (e.g. low-dimensional bottleneck between residual and decoder, or few tokens). The model must compress; it often keeps **semantic** information and drops texture.

- **Disentanglement**: Encourage **independence** across tokens or dimensions (e.g. correlation loss, or total correlation). Sometimes yields more interpretable (and semantic) factors (e.g. "object presence", "position").

---

## 6. Summary: what might work best for you

| Idea | Why it can help (no labels) |
|------|-----------------------------|
| **Sparse context recon** | Model must use prior; prior is structure/object. |
| **Sparse latent usage** | Tokens specialize → object vs background, parts. |
| **Multi-crop / scale consistency** | Augmentation structure gives "same content" signal. |
| **Edge/structure auxiliary** | Boundaries = object vs background; no labels. |
| **Deep clustering** | Pseudo-labels from clustering can refine semantics. |

**Practical order to try**: (1) **Sparse context** reconstruction (e.g. 20% random pixels as context, reconstruct full image). (2) **Sparse attention** over latent tokens (low temperature or L1). (3) **Edge auxiliary** loss. (4) **Multi-crop** or scale consistency on φ. (5) **Deep clustering** on top of current NCE features.

If you want, we can implement one of these (e.g. sparse context recon or sparse token usage) in your current OmniField + NCE setup.
