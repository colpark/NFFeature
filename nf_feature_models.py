# Shared model and helpers for NFFeature (OmniField-style CNF)
# Used by AblationCIFAR10.ipynb and TDSM_Classification.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi, log
from functools import wraps
from einops import rearrange, repeat

D = 256
# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.latest_attn = None

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim = -1)
        self.latest_attn = attn.detach()
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

from math import log

# This helper function creates the sinusoidal embeddings
def get_sinusoidal_embeddings(n, d):
    """
    Generates sinusoidal positional embeddings.
    
    Args:
        n (int): The number of positions (num_latents).
        d (int): The embedding dimension (latent_dim).

    Returns:
        torch.Tensor: A tensor of shape (n, d) with sinusoidal embeddings.
    """
    # Ensure latent_dim is even for sin/cos pairs
    assert d % 2 == 0, "latent_dim must be an even number for sinusoidal embeddings"
    
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(log(10000.0) / d))
    
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def add_white_noise(coords, scale=0.01):
    return coords + torch.randn_like(coords) * scale




class CascadedBlock(nn.Module):
    def __init__(self, dim, n_latents, input_dim, cross_heads, cross_dim_head, self_heads, self_dim_head, residual_dim=None):
        super().__init__()
        self.latents = nn.Parameter(get_sinusoidal_embeddings(n_latents, dim), requires_grad=False)
        # self.latents = nn.Parameter(torch.randn(n_latents, dim))
        self.cross_attn = PreNorm(dim, Attention(dim, input_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=input_dim)
        self.self_attn = PreNorm(dim, Attention(dim, heads=self_heads, dim_head=self_dim_head))
        self.residual_proj = nn.Linear(residual_dim, dim) if residual_dim and residual_dim != dim else None
        self.ff = PreNorm(dim, FeedForward(dim))

    def forward(self, x, context, mask=None, residual=None):
        b = context.size(0)
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        latents = self.cross_attn(latents, context=context, mask=mask) + latents
        if residual is not None:
            if self.residual_proj:
                residual = self.residual_proj(residual)
            latents = latents + residual
        latents = self.self_attn(latents) + latents
        latents = self.ff(latents) + latents
        return latents


class CascadedPerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        queries_dim,
        logits_dim = None,
        latent_dims=(512, 512, 512),
        num_latents=(256, 256, 256),
        cross_heads = 4,
        cross_dim_head = 128,
        self_heads = 8,
        self_dim_head = 128,
        decoder_ff = False,
        
    ):
        super().__init__()
        
        assert len(latent_dims) == len(num_latents), "latent_dims and num_latents must have same length"
        
    
        # self.input_proj = nn.Linear(4, 128)
        self.input_proj = nn.Sequential(
                nn.Linear(4, 128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        self.projection_matrix = nn.Parameter(torch.randn(4, 128) / np.sqrt(4))
        # proj = torch.randn(4, 128) / np.sqrt(4)
        # self.projection_matrix = nn.Parameter(proj.detach())  # make it a leaf tenso

        # Cascaded encoder blocks
        self.encoder_blocks = nn.ModuleList()
        prev_dim = None
        for dim, n_latents in zip(latent_dims, num_latents):
            block = CascadedBlock(
                dim=dim,
                n_latents=n_latents,
                input_dim=input_dim,
                cross_heads=cross_heads,
                cross_dim_head=cross_dim_head,
                self_heads=self_heads,
                self_dim_head=self_dim_head,
                residual_dim=prev_dim
            )
            self.encoder_blocks.append(block)
            prev_dim = dim

        # Decoder
        final_latent_dim = latent_dims[-1]
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=final_latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        

        # self.decoder_swin = SwinTransformerLayer(
        #     dim=queries_dim,
        #     depth=2,                  # or 4 if you want deeper decoding
        #     num_heads=4,
        #     window_size=16,           # assuming 64x64 → 4096 tokens → 256 windows of size 16
        #     mlp_ratio=4.0,
        #     drop_path=0.1,
        #     use_checkpoint=False
        # )
        self.self_attn_blocks = nn.Sequential(*[
        nn.Sequential(
            PreNorm(latent_dims[-1], Attention(latent_dims[-1], heads=self_heads, dim_head=self_dim_head)),
            PreNorm(latent_dims[-1], FeedForward(latent_dims[-1]))
        )
        for _ in range(4)  # or 3
    ])

    def forward(self, data, mask=None, queries=None):
        b = data.size(0)
        residual = None

        
        for block in self.encoder_blocks:
            residual = block(x=residual, context=data, mask=mask, residual=residual)

            
            
            
        for sa_block in self.self_attn_blocks:
            residual = sa_block[0](residual) + residual
            residual = sa_block[1](residual) + residual
        
        if  b == 1:  # Optional: only log for one sample
            latent_std = residual.std(dim=1).mean().item()
            print(f"[Latent std]: {latent_std:.4f}")
        
        if queries is None:
            return latents

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)

        x = self.decoder_cross_attn(queries, context=residual)

        # Optional: skip connection to preserve input query encoding
        x = x + queries

        # Local refinement (like SCENT)
        # x = self.decoder_swin(x)

        # Final FF
        if self.decoder_ff:
            x = x + self.decoder_ff(x)

        return self.to_logits(x)


class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn((in_features, mapping_size)) * scale)

    def forward(self, coords):
        projections = coords @ self.B
        fourier_feats = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return fourier_feats


def create_coordinate_grid(h, w, device):
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing='ij'
    ), dim=-1)
    return rearrange(grid, 'h w c -> (h w) c')



def prepare_model_input(images, coords, fourier_encoder_fn):
    b, c, h, w = images.shape
    pixels = rearrange(images, 'b c h w -> b (h w) c')
    batch_coords = repeat(coords, 'n d -> b n d', b=b)
    pos_embeddings = fourier_encoder_fn(batch_coords)
    input_with_pos = torch.cat((pixels, pos_embeddings), dim=-1)
    return input_with_pos, pixels, pos_embeddings

