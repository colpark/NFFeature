"""
Soft InfoNCE extra visualizations — paste into a running session (e.g. after the
"Coordinate error" cell in SoftInfoNCE_OmniField_CIFAR10.ipynb).

Expects in scope:
  imgs, imgs_b, R, t, anchors_a, coords_b_batch, xi_mapped, w, logits,
  pred_coords, b_show, n_anchors_show, cfg, IMAGE_SIZE, plt, torch, np
If pred_coords not yet computed, run first:
  best_j = logits[b_show].argmax(dim=1)
  pred_coords = coords_b_batch[b_show][best_j]
"""

import numpy as np
import matplotlib.pyplot as plt

def norm_to_pixel(coords_norm, h, w):
    y, x = coords_norm[..., 0], coords_norm[..., 1]
    row = (y + 1) / 2 * (h - 1)
    col = (x + 1) / 2 * (w - 1)
    return col.cpu().numpy(), row.cpu().numpy()


# ----- 1. View A vs View B: anchors on A, GT and predicted match on B -----
n_show = min(8, anchors_a.size(1))
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(imgs[b_show].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
axs[0].set_title('View A (anchors)')
axs[0].axis('off')
cx_a, cy_a = norm_to_pixel(anchors_a[b_show, :n_show], IMAGE_SIZE, IMAGE_SIZE)
axs[0].scatter(cx_a, cy_a, c='lime', s=40, marker='o', edgecolors='black', linewidths=0.5, label='anchors')

axs[1].imshow(imgs_b[b_show].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
axs[1].set_title('View B (GT vs predicted match)')
axs[1].axis('off')
cx_gt, cy_gt = norm_to_pixel(xi_mapped[b_show, :n_show], IMAGE_SIZE, IMAGE_SIZE)
cx_pr, cy_pr = norm_to_pixel(pred_coords[:n_show], IMAGE_SIZE, IMAGE_SIZE)
axs[1].scatter(cx_gt, cy_gt, c='lime', s=60, marker='+', linewidths=2, label='GT T(x)')
axs[1].scatter(cx_pr, cy_pr, c='cyan', s=40, marker='x', linewidths=1.5, label='pred (argmax)')
axs[1].legend(loc='upper right', fontsize=8)
plt.suptitle('Correspondence: green = anchor/GT, cyan = model prediction')
plt.tight_layout()
plt.savefig('softnce_viewA_viewB.png', dpi=100)
plt.show()


# ----- 2. Retrieval @ ε -----
eps_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
err_all = (pred_coords - xi_mapped[b_show]).norm(dim=-1).cpu().numpy()
acc_at_eps = [(err_all < eps).mean() * 100 for eps in eps_values]
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(len(eps_values)), acc_at_eps, color='steelblue', edgecolor='black')
ax.set_xticks(range(len(eps_values)))
ax.set_xticklabels([str(e) for e in eps_values])
ax.set_xlabel('ε (normalized coord distance)')
ax.set_ylabel('% anchors with error < ε')
ax.set_title('Retrieval accuracy: how often does argmax fall near GT?')
plt.tight_layout()
plt.savefig('softnce_retrieval_at_eps.png', dpi=100)
plt.show()
print('Retrieval @ 0.1: {:.1f}%  @ 0.2: {:.1f}%'.format(acc_at_eps[1], acc_at_eps[3]))


# ----- 3. Soft weight concentration -----
w_max = w.max(dim=2).values.flatten().cpu().numpy()
plt.figure(figsize=(5, 3))
plt.hist(w_max, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
plt.xlabel('max_j w_ij (soft weight on best candidate)')
plt.ylabel('Count (anchors)')
plt.title('Concentration: peaked weights = confident correspondence')
plt.axvline(w_max.mean(), color='red', linestyle='--', label=f'mean={w_max.mean():.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('softnce_weight_concentration.png', dpi=100)
plt.show()


# ----- 4. Heatmap overlay: soft weights with GT (+) and predicted (x) -----
fig, axs = plt.subplots(2, n_anchors_show, figsize=(12, 5))
for i in range(n_anchors_show):
    heat = w[b_show, i].cpu().numpy().reshape(IMAGE_SIZE, IMAGE_SIZE)
    axs[0, i].imshow(heat, cmap='hot')
    axs[0, i].set_title(f'Anchor {i}')
    cx_gt, cy_gt = norm_to_pixel(xi_mapped[b_show, i:i+1], IMAGE_SIZE, IMAGE_SIZE)
    cx_pr, cy_pr = norm_to_pixel(pred_coords[i:i+1], IMAGE_SIZE, IMAGE_SIZE)
    axs[0, i].scatter(cx_gt, cy_gt, c='lime', s=80, marker='+', linewidths=2, label='GT')
    axs[0, i].scatter(cx_pr, cy_pr, c='cyan', s=50, marker='x', linewidths=1.5, label='pred')
    axs[0, i].axis('off')
    axs[1, i].imshow(imgs_b[b_show].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axs[1, i].scatter(cx_gt, cy_gt, c='lime', s=80, marker='+', linewidths=2)
    axs[1, i].scatter(cx_pr, cy_pr, c='cyan', s=50, marker='x', linewidths=1.5)
    axs[1, i].axis('off')
plt.suptitle('Soft weights: lime = GT T(x), cyan = argmax prediction')
plt.tight_layout()
plt.savefig('softnce_heatmap_overlay.png', dpi=100)
plt.show()


# ========== Feature-level difference: what NCE learned ==========
# Expects: phi_raw_a, phi_raw_b, ProjectionHead, QUERIES_DIM, PROJ_DIM, DEVICE

# Cosine similarity (logits * tau = cos sim), GT match index per anchor
S_nce = (logits[b_show] * cfg['tau']).detach().cpu().numpy()
N_a, N_b = S_nce.shape
sqd_b = ((coords_b_batch[b_show].unsqueeze(0) - xi_mapped[b_show].unsqueeze(1)) ** 2).sum(-1).cpu().numpy()
j_gt = np.argmin(sqd_b, axis=1)
pos_sims = S_nce[np.arange(N_a), j_gt]
neg_mask = np.ones((N_a, N_b), dtype=bool)
neg_mask[np.arange(N_a), j_gt] = False
neg_sims = S_nce[neg_mask]

# Pos vs neg histogram (NCE)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(neg_sims, bins=40, alpha=0.6, color='coral', label='negative pairs', density=True)
ax.hist(pos_sims, bins=30, alpha=0.6, color='green', label='positive pairs (GT corr.)', density=True)
ax.axvline(pos_sims.mean(), color='green', linestyle='--', linewidth=1.5, label=f'pos mean={pos_sims.mean():.3f}')
ax.axvline(neg_sims.mean(), color='coral', linestyle='--', linewidth=1.5, label=f'neg mean={neg_sims.mean():.3f}')
ax.set_xlabel('Cosine similarity φ(anchor) · φ(candidate)')
ax.set_ylabel('Density')
ax.set_title('Feature-level: NCE-trained (pos vs neg)')
ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('softnce_feature_pos_neg_hist.png', dpi=100)
plt.show()
print(f'Positive mean: {pos_sims.mean():.4f}  Negative mean: {neg_sims.mean():.4f}  Gap: {pos_sims.mean()-neg_sims.mean():.4f}')

# Margin
margin_nce = pos_sims - S_nce.mean(axis=1)
plt.figure(figsize=(5, 3))
plt.hist(margin_nce, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
plt.axvline(margin_nce.mean(), color='red', linestyle='--', label=f'mean margin={margin_nce.mean():.3f}')
plt.xlabel('Margin (sim to GT − mean sim to candidates)')
plt.ylabel('Count')
plt.title('Feature margin: how much higher is sim to true correspondence?')
plt.legend()
plt.tight_layout()
plt.savefig('softnce_feature_margin.png', dpi=100)
plt.show()

# Baseline (random proj) vs NCE
import torch
proj_baseline = ProjectionHead(QUERIES_DIM, PROJ_DIM).to(DEVICE)
with torch.no_grad():
    phi_bl_a = proj_baseline(phi_raw_a)
    phi_bl_b = proj_baseline(phi_raw_b)
S_bl = (torch.bmm(phi_bl_a, phi_bl_b.transpose(1, 2))[b_show]).cpu().numpy()
pos_sims_bl = S_bl[np.arange(N_a), j_gt]
neg_sims_bl = S_bl[neg_mask]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(neg_sims, bins=40, alpha=0.4, color='coral', label='NCE neg', density=True)
ax.hist(pos_sims, bins=30, alpha=0.5, color='green', label='NCE pos', density=True)
ax.hist(neg_sims_bl, bins=40, alpha=0.4, color='gray', histtype='step', linewidth=2, label='baseline neg', density=True)
ax.hist(pos_sims_bl, bins=30, alpha=0.5, color='blue', histtype='step', linewidth=2, label='baseline pos', density=True)
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title('Feature-level: NCE (filled) vs baseline / random proj (outline)')
ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('softnce_feature_baseline_vs_nce.png', dpi=100)
plt.show()
print('NCE:      pos mean={:.4f}  neg mean={:.4f}  gap={:.4f}'.format(pos_sims.mean(), neg_sims.mean(), pos_sims.mean()-neg_sims.mean()))
print('Baseline: pos mean={:.4f}  neg mean={:.4f}  gap={:.4f}'.format(pos_sims_bl.mean(), neg_sims_bl.mean(), pos_sims_bl.mean()-neg_sims_bl.mean()))


# ========== TDSM: token-decoded spatial map (same as TDSM_Classification.ipynb), baseline vs NCE ==========
# One latent token -> full 32x32 decoded image. Baseline is texture-like; does NCE change it?
# Expects: model, fourier_encoder, get_residual, prepare_model_input, coords_32, val_loader, DEVICE,
#          IMAGE_SIZE, INPUT_DIM, QUERIES_DIM, LOGITS_DIM, FOURIER_MAPPING_SIZE, torch
# Uses: repeat (einops) if available, else fallback.
try:
    baseline_model
except NameError:
    import os
    CKPT_PATH = os.path.join('checkpoints', 'checkpoint_best.pt')
    if not os.path.isfile(CKPT_PATH):
        CKPT_PATH = os.path.join('checkpoints', 'checkpoint_last.pt')
    from nf_feature_models import CascadedPerceiverIO, GaussianFourierFeatures
    baseline_fourier = GaussianFourierFeatures(in_features=2, mapping_size=FOURIER_MAPPING_SIZE, scale=15.0).to(DEVICE)
    baseline_model = CascadedPerceiverIO(
        input_dim=INPUT_DIM, queries_dim=QUERIES_DIM, logits_dim=LOGITS_DIM,
        latent_dims=(256, 384, 512), num_latents=(256, 256, 256), decoder_ff=True,
    ).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    baseline_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    baseline_fourier.load_state_dict(ckpt['fourier_encoder_state_dict'], strict=False)
    baseline_model.eval()
    baseline_fourier.eval()
    for p in list(baseline_model.parameters()) + list(baseline_fourier.parameters()):
        p.requires_grad = False
    print('Baseline model (no NCE) loaded from', CKPT_PATH)

def decoder_forward(model, queries, context):
    x = model.decoder_cross_attn(queries, context=context)
    x = x + queries
    if getattr(model, 'decoder_ff', None) is not None:
        x = x + model.decoder_ff(x)
    return model.to_logits(x)

def get_tdsm(model, fourier_encoder, data, coords_32, device, num_tokens=256, token_step=4):
    from einops import repeat
    with torch.no_grad():
        residual = get_residual(model, data)
        B = data.size(0)
        queries_32 = fourier_encoder(repeat(coords_32, 'n d -> b n d', b=B)).to(device)
        component_images = []
        for k in range(0, num_tokens, token_step):
            ctx_k = residual[:, k:k+1, :]
            logits_k = decoder_forward(model, queries_32, ctx_k)
            img_k = logits_k.reshape(B, IMAGE_SIZE, IMAGE_SIZE, 3)
            component_images.append(img_k)
        component_images = torch.stack(component_images, dim=1)
        tdsm = component_images.mean(dim=-1)
    return tdsm

TDSM_TOKEN_STEP = 4
imgs_tdsm, _ = next(iter(val_loader))
imgs_tdsm = imgs_tdsm[:4].to(DEVICE)
input_tdsm, _, _ = prepare_model_input(imgs_tdsm, coords_32, fourier_encoder)
with torch.no_grad():
    tdsm_baseline = get_tdsm(baseline_model, baseline_fourier, input_tdsm, coords_32, DEVICE, token_step=TDSM_TOKEN_STEP)
    tdsm_nce      = get_tdsm(model, fourier_encoder, input_tdsm, coords_32, DEVICE, token_step=TDSM_TOKEN_STEP)

sample_idx = 0
token_indices = [0, 16, 32, 48]
tdsm_slice_idx = [k // TDSM_TOKEN_STEP for k in token_indices]
n_show = len(token_indices)
fig, axs = plt.subplots(2, n_show, figsize=(12, 5))
for i, (k, sk) in enumerate(zip(token_indices, tdsm_slice_idx)):
    axs[0, i].imshow(tdsm_baseline[sample_idx, sk].cpu().numpy(), cmap='viridis')
    axs[0, i].set_title('Token %d baseline' % k)
    axs[0, i].axis('off')
    axs[1, i].imshow(tdsm_nce[sample_idx, sk].cpu().numpy(), cmap='viridis')
    axs[1, i].set_title('Token %d NCE' % k)
    axs[1, i].axis('off')
plt.suptitle('TDSM: per-token recon (baseline = texture-like; does NCE change structure?)')
plt.tight_layout()
plt.savefig('softnce_tdsm_baseline_vs_nce.png', dpi=100)
plt.show()

# RGB component images for same tokens
from einops import repeat
with torch.no_grad():
    residual_bl = get_residual(baseline_model, input_tdsm)
    residual_nce = get_residual(model, input_tdsm)
    queries_32 = fourier_encoder(repeat(coords_32, 'n d -> b n d', b=imgs_tdsm.size(0))).to(DEVICE)
    comps_bl, comps_nce = [], []
    for k in token_indices:
        comps_bl.append(decoder_forward(baseline_model, queries_32, residual_bl[:, k:k+1, :]).reshape(imgs_tdsm.size(0), IMAGE_SIZE, IMAGE_SIZE, 3))
        comps_nce.append(decoder_forward(model, queries_32, residual_nce[:, k:k+1, :]).reshape(imgs_tdsm.size(0), IMAGE_SIZE, IMAGE_SIZE, 3))
    comps_bl = torch.stack(comps_bl, dim=0)
    comps_nce = torch.stack(comps_nce, dim=0)
def to_display(t):
    return (t.cpu() / 2 + 0.5).clamp(0, 1) if t.abs().max() > 1.5 else t.cpu().clamp(0, 1)
fig, axs = plt.subplots(2, n_show + 1, figsize=(14, 5))
axs[0, 0].imshow(to_display(imgs_tdsm[sample_idx]).permute(1, 2, 0).numpy())
axs[0, 0].set_title('Input')
axs[0, 0].axis('off')
axs[1, 0].axis('off')
for i in range(n_show):
    axs[0, i+1].imshow(to_display(comps_bl[i, sample_idx]).numpy())
    axs[0, i+1].set_title('Token %d baseline' % token_indices[i])
    axs[0, i+1].axis('off')
    axs[1, i+1].imshow(to_display(comps_nce[i, sample_idx]).numpy())
    axs[1, i+1].set_title('Token %d NCE' % token_indices[i])
    axs[1, i+1].axis('off')
plt.suptitle('Per-token RGB component: baseline vs NCE (texture vs structure?)')
plt.tight_layout()
plt.savefig('softnce_tdsm_components_baseline_vs_nce.png', dpi=100)
plt.show()


# ========== TDSM class/semantics: t-SNE and PCA colored by CIFAR-10 class ==========
N_VAL_TDSM = min(500, len(val_loader.dataset))
all_feat_baseline, all_feat_nce, all_labels = [], [], []
n_done = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        if n_done >= N_VAL_TDSM:
            break
        imgs = imgs.to(DEVICE)
        input_data, _, _ = prepare_model_input(imgs, coords_32, fourier_encoder)
        tdsm_bl = get_tdsm(baseline_model, baseline_fourier, input_data, coords_32, DEVICE, token_step=TDSM_TOKEN_STEP)
        tdsm_n = get_tdsm(model, fourier_encoder, input_data, coords_32, DEVICE, token_step=TDSM_TOKEN_STEP)
        all_feat_baseline.append(tdsm_bl.mean(dim=(2, 3)).cpu().numpy())
        all_feat_nce.append(tdsm_n.mean(dim=(2, 3)).cpu().numpy())
        all_labels.append(labels.numpy())
        n_done += imgs.size(0)
X_baseline = np.concatenate(all_feat_baseline, axis=0)[:N_VAL_TDSM]
X_nce = np.concatenate(all_feat_nce, axis=0)[:N_VAL_TDSM]
y_all = np.concatenate(all_labels, axis=0)[:N_VAL_TDSM]
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    n_comp = min(50, X_baseline.shape[1], X_baseline.shape[0] - 1)
    X_bl_pca = PCA(n_components=n_comp).fit_transform(X_baseline)
    X_nce_pca = PCA(n_components=n_comp).fit_transform(X_nce)
    X_bl_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_bl_pca)
    X_nce_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_nce_pca)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X_2d, title in [(axs[0], X_bl_tsne, 'Baseline (no NCE)'), (axs[1], X_nce_tsne, 'NCE-trained')]:
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, cmap='tab10', s=12, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=axs, label='Class', shrink=0.6)
    plt.suptitle('TDSM features (pooled): t-SNE colored by CIFAR-10 class')
    plt.tight_layout()
    plt.savefig('softnce_tdsm_tsne_class.png', dpi=100)
    plt.show()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X_2d, title in [(axs[0], X_bl_pca[:, :2], 'Baseline PCA'), (axs[1], X_nce_pca[:, :2], 'NCE PCA')]:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, cmap='tab10', s=12, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.suptitle('TDSM features: PCA first 2 components (class-colored)')
    plt.tight_layout()
    plt.savefig('softnce_tdsm_pca_class.png', dpi=100)
    plt.show()
except ImportError as e:
    print('Install sklearn for t-SNE/PCA:', e)


# ========== What changed: object vs background ==========
# Spatial difference (mean over tokens |NCE - Baseline|)
diff_spatial = (tdsm_nce - tdsm_baseline).abs().mean(dim=1).cpu().numpy()
fig, axs = plt.subplots(2, 4, figsize=(14, 6))
for i in range(4):
    axs[0, i].imshow(imgs_tdsm[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axs[0, i].set_title('Input' if i == 0 else ''); axs[0, i].axis('off')
    im = axs[1, i].imshow(diff_spatial[i], cmap='hot')
    axs[1, i].set_title('|NCE−Baseline| (mean over tokens)' if i == 0 else ''); axs[1, i].axis('off')
plt.colorbar(im, ax=axs[1, :], shrink=0.6, label='Mean |diff|')
plt.suptitle('Where NCE changed TDSM: object vs background')
plt.tight_layout()
plt.savefig('softnce_tdsm_spatial_diff.png', dpi=100)
plt.show()

# Per-token object sensitivity (center vs border)
H, W = IMAGE_SIZE, IMAGE_SIZE
margin = 8
obj_mask = np.zeros((H, W), dtype=np.float32)
obj_mask[margin:H-margin, margin:W-margin] = 1.0
bg_mask = 1.0 - obj_mask
obj_mask = torch.from_numpy(obj_mask).to(DEVICE).view(1, 1, H, W)
bg_mask = torch.from_numpy(bg_mask).to(DEVICE).view(1, 1, H, W)
n_tokens = tdsm_baseline.size(1)
obj_bl = (tdsm_baseline * obj_mask).sum(dim=(2, 3)) / (obj_mask.sum() + 1e-8)
bg_bl = (tdsm_baseline * bg_mask).sum(dim=(2, 3)) / (bg_mask.sum() + 1e-8)
obj_nce = (tdsm_nce * obj_mask).sum(dim=(2, 3)) / (obj_mask.sum() + 1e-8)
bg_nce = (tdsm_nce * bg_mask).sum(dim=(2, 3)) / (bg_mask.sum() + 1e-8)
sens_bl = (obj_bl - bg_bl).mean(dim=0).cpu().numpy()
sens_nce = (obj_nce - bg_nce).mean(dim=0).cpu().numpy()
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].bar(np.arange(n_tokens), sens_bl, color='steelblue', alpha=0.8); axs[0].set_xlabel('Token index'); axs[0].set_ylabel('Object sensitivity'); axs[0].set_title('Baseline')
axs[1].bar(np.arange(n_tokens), sens_nce, color='green', alpha=0.8); axs[1].set_xlabel('Token index'); axs[1].set_ylabel('Object sensitivity'); axs[1].set_title('NCE')
plt.suptitle('Per-token object sensitivity (center − border)')
plt.tight_layout()
plt.savefig('softnce_tdsm_object_sensitivity.png', dpi=100)
plt.show()
plt.figure(figsize=(5, 5))
plt.scatter(sens_bl, sens_nce, alpha=0.7)
plt.plot([sens_bl.min(), sens_bl.max()], [sens_bl.min(), sens_bl.max()], 'r--', label='y=x')
plt.xlabel('Baseline object sensitivity'); plt.ylabel('NCE object sensitivity')
plt.title('Per-token: above line = NCE more object-focused')
plt.legend(); plt.tight_layout()
plt.savefig('softnce_tdsm_sensitivity_scatter.png', dpi=100)
plt.show()
