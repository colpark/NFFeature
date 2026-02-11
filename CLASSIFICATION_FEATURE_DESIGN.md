# Using the Neural Field for Classification: Token-Decoded Decomposition

## Your idea (in one sentence)

For each **learnable latent token** from the processor, run the **decoder once** (with that single token as context) to get a **full 32×32×3 “component” image**; stack these into a **spatial feature map** of shape **(32, 32, num_tokens)** (or with channels), and use this **“soup” of data-driven decompositions** as the input to a classifier.

---

## 1. Naming

### Approach name

- **Token-wise decoded decomposition (TWD)**  
  “We decode the field **per token** to get token-wise reconstructions, then use them as a decomposition of the input.”

- **Latent component decoding (LCD)**  
  Each latent is treated as one “component”; decoding it gives one component image.

- **Per-token field decoding (PTFD)**  
  One forward per token through the decoder to get a field (image) per token.

**Recommendation:** **Token-wise decoded decomposition (TWD)** — clear and descriptive.

### Feature name

The resulting representation is:

- A **spatial map** (32×32) where at each location you have a **vector of length num_tokens** (one value per token’s decoded image at that pixel).
- So it’s a “soup” of **data-dependent components** (each token’s contribution to the reconstruction at every pixel).

Possible names for this feature:

| Name | Rationale |
|------|-----------|
| **Token-decoded spatial map (TDSM)** | Emphasizes: from tokens, decoded, spatial. |
| **Decomposed field features (DFF)** | Fits NFFeature; “field” = neural field. |
| **Latent soup features (LSF)** | “Soup” = mixture of token contributions. |
| **Token component map (TCM)** | One “component” image per token, then stacked into a map. |

**Recommendation:** **Token-decoded spatial map (TDSM)** for the 32×32×num_tokens tensor; when you pool it for a classifier you can say “TDSM-pooled features” (e.g. “TDSM with global average pooling”).

---

## 2. Precise formulation

### What you have after the processor

- **residual** = latent field: shape `(B, num_latents, latent_dim)`  
  - e.g. `(B, 256, 512)`.
- Decoder: `queries` `(B, N_q, queries_dim)` × `context` `(B, num_latents, latent_dim)` → `(B, N_q, 3)`.

### Per-token decoding

- **For each token index** `k = 0 .. num_latents - 1`:
  - Set **context_k** = `residual[:, k:k+1, :]`  → `(B, 1, latent_dim)`.
  - **queries** = GFF at full 32×32 grid → `(B, 1024, queries_dim)`.
  - **out_k** = Decoder(queries, context=context_k) → `(B, 1024, 3)`.
  - Reshape: **I_k** = `(B, 32, 32, 3)` — “component image” for token `k`.

- Stack over `k`: **component_images** = `(B, num_latents, 32, 32, 3)`.

### Build the classification feature (32×32×num_tokens)

You want “32 × 32 × learnable tokens”:

- At each spatial position `(i, j)`, you want one scalar (or vector) **per token**.
- From **I_k** at `(i,j)` you have 3 values (RGB). Options:
  - **Option A (scalar per token):** e.g. mean over RGB:  
    `feature[b, i, j, k] = I_k[b, i, j, :].mean()`  
    → **TDSM** shape `(B, 32, 32, num_tokens)`.
  - **Option B (keep RGB per token):**  
    `feature` = `(B, 32, 32, num_tokens * 3)` — “TDSM-RGB”.
  - **Option C (L2 norm per token):**  
    `feature[b,i,j,k] = ||I_k[b,i,j,:]||_2` — emphasizes magnitude of that component at that location.

So:

- **TDSM** = token-decoded spatial map = **(32, 32, num_tokens)** (one scalar per token per pixel, e.g. mean over RGB).
- This is the “soup of original data feature decomposition”: each token contributes one “slice” of the decomposition over space.

---

## 3. How to achieve it (implementation strategies)

### 3.1 Naive loop (clear, slower)

```python
# residual: (B, num_latents, latent_dim), e.g. (B, 256, 512)
# queries_32: (B, 1024, queries_dim) for 32x32 grid
B, num_latents, latent_dim = residual.shape
component_images = []  # list of (B, 1024, 3)
for k in range(num_latents):
    ctx_k = residual[:, k:k+1, :]  # (B, 1, latent_dim)
    out_k = model.decode(queries_32, context=ctx_k)  # (B, 1024, 3)
    component_images.append(out_k)
# stack -> (B, num_latents, 1024, 3) -> (B, num_latents, 32, 32, 3)
component_images = torch.stack(component_images, dim=1)
# TDSM: (B, 32, 32, num_tokens) by mean over RGB
tdsm = component_images.mean(dim=-1).permute(0, 2, 3, 1)  # (B, 32, 32, num_latents)
```

- **Pros:** Easiest to implement; reuses existing decoder; no change to model.
- **Cons:** 256 decoder forwards per sample (or per batch). Can be batched over tokens with a loop; still 256 steps.

### 3.2 Batched “per-token” decoder (one big forward)

The current decoder does:  
`cross_attn(queries, context=residual)` → one output per query.  
So with `context = (B, 256, 512)` you get one (query-dependent) output, not 256 separate outputs.

To get **per-token** outputs in one shot you’d need to change the decoder or add a wrapper:

- **Variant A — Loop over tokens, batch over B:**  
  For each `k`, run decoder for **all** batch elements at once:  
  `context_k = residual[:, k:k+1, :]` → `(B, 1, latent_dim)`.  
  One decoder forward gives `(B, 1024, 3)`. So you still need **num_latents** forwards, but each forward is over the full batch. Same as 3.1 but written as a loop over `k`; no extra memory for 256 copies of the decoder.

- **Variant B — True batched per-token in one forward:**  
  Reshape so the decoder sees “batch = B × num_latents”:
  - `residual_expanded`: treat each (batch, token) as a batch item:  
    `(B, num_latents, latent_dim)` → `(B * num_latents, 1, latent_dim)`.
  - `queries` repeated for each token:  
    `queries_32`: `(B, 1024, queries_dim)` → repeat for each token → `(B * num_latents, 1024, queries_dim)`.
  - One decoder forward: `(B*num_latents, 1024, queries_dim)` × `(B*num_latents, 1, latent_dim)` → need to implement decoder so it accepts (B*num_latents) and doesn’t mix them. Actually the decoder does cross_attn(queries, context); if batch is (B*num_latents), then for each of B*num_latents we have context (1, latent_dim) and queries (1024, queries_dim). So we’d run decoder with batch size B*num_latents, context (B*num_latents, 1, latent_dim), queries (B*num_latents, 1024, queries_dim). Output (B*num_latents, 1024, 3). Reshape to (B, num_latents, 1024, 3). **One forward**, but batch size becomes B*256 — heavy on memory.

So:

- **3.1 / 3.2A:** 256 decoder forwards, batch size B; **simplest and likely best first step.**
- **3.2B:** One forward with batch B*256; **faster but high memory**; need to ensure decoder and data loaders support large batch.

### 3.3 Reduce the number of tokens for classification (compromise)

- Use only a **subset** of tokens (e.g. every 4th, or 64 tokens) to build TDSM.
- Then TDSM shape is (32, 32, 64) — smaller and fewer decoder passes (e.g. 64 instead of 256).
- You can justify this as “a sparse decomposition” or “principal components” if you add a small learned selection later.

---

## 4. From TDSM to classification

You have **tdsm** of shape `(B, 32, 32, num_tokens)`.

Options:

1. **Global average pool over space:**  
   `feat = tdsm.mean(dim=(1,2))` → `(B, num_tokens)`.  
   Then a linear layer: `num_tokens → num_classes`.  
   Fast and simple; treats each token as one global feature.

2. **Global max pool:**  
   `feat = tdsm.max(dim=1).values.max(dim=1).values` → `(B, num_tokens)`.

3. **Flatten spatial:**  
   `feat = tdsm.flatten(1)` → `(B, 32*32*num_tokens)`.  
   Very large; usually need a linear or MLP to reduce before classification.

4. **Small CNN on TDSM:**  
   Treat (32, 32, num_tokens) as a multi-channel image; 1×1 or 3×3 convs then pool then linear.  
   Lets the classifier learn which spatial regions and which tokens matter.

5. **Attention over tokens (at each location or global):**  
   Learn a weight over tokens from TDSM, then pool. E.g. `attn = softmax(linear(tdsm))` then `feat = (tdsm * attn).sum(dim=-1)` then spatial pool.

**Recommendation:** Start with **global average pool** → `(B, num_tokens)` → linear → classes. If that works, try a tiny CNN on TDSM or attention over tokens.

---

## 5. Training strategy

- **Option A — Freeze backbone, train only classifier:**  
  Extract TDSM with the pretrained (reconstruction-trained) model fixed; train a linear (or MLP) on top. Fast; tests whether TDSM is already discriminative.

- **Option B — Fine-tune backbone + classifier:**  
  Backprop through the decoder and processor (and optionally encoder) into the TDSM and then the classifier. More capacity; risk of forgetting reconstruction if you don’t use a combined loss.

- **Option C — Two-stage:**  
  Train classifier on frozen TDSM first; then optionally unfreeze and fine-tune with a small LR and maybe a combined loss (reconstruction + classification).

---

## 6. Summary table

| Item | Recommendation |
|------|----------------|
| **Approach name** | **Token-wise decoded decomposition (TWD)** |
| **Feature name** | **Token-decoded spatial map (TDSM)**; shape (32, 32, num_tokens) |
| **Per-token scalar** | Mean over RGB of that token’s decoded image at each pixel |
| **Implementation** | Loop over tokens, one decoder forward per token (context = that token only); then stack and reduce to TDSM |
| **Classification** | TDSM → global average pool → (B, num_tokens) → linear → num_classes |
| **Training** | Start with frozen backbone, train classifier only |

---

## 7. Optional: helper API

You can add a method to the model (or a standalone function):

```python
def get_tdsm(self, data, queries_32x32, reduce_rgb='mean'):
    """
    Token-decoded spatial map for classification.
    - data: (B, N, input_dim) context
    - queries_32x32: (B, 1024, queries_dim)
    Returns: tdsm (B, 32, 32, num_latents)
    """
    residual = self.forward_processor_only(data)  # or run full forward but stop before decoder
    # loop over tokens, decode each -> component_images (B, num_latents, 32, 32, 3)
    # then reduce to (B, 32, 32, num_latents)
    ...
```

And a small classifier module:

```python
class TDSMClassifier(nn.Module):
    def __init__(self, num_tokens, num_classes, pool='avg'):
        ...
    def forward(self, tdsm):
        # tdsm (B, 32, 32, num_tokens)
        if self.pool == 'avg':
            x = tdsm.mean(dim=(1, 2))  # (B, num_tokens)
        return self.fc(x)
```

This keeps the “TWD + TDSM” pipeline clear and reusable.
