from typing import List, Optional, Tuple, Dict, Any
import math
import mindspore as ms

from mindspore import mint, nn

class AdaLayerNormZero(nn.Cell):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_in_dim: int, embedding_out_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = mint.nn.Linear(embedding_in_dim, 6 * embedding_out_dim, bias=True)

        self.norm = mint.nn.LayerNorm(embedding_out_dim, elementwise_affine=False, eps=1e-6)

    def construct(
        self,
        x: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TransformerBlock(nn.Cell):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = mint.nn.LayerNorm(d_model)
        self.attn = ms.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = mint.nn.Dropout(dropout)
        self.norm2 = mint.nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(d_model, hidden),
            nn.SiLU(),
            mint.nn.Dropout(dropout),
            mint.nn.Linear(hidden, d_model),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.dropout(h)
        return x
    
class Conv1DHead(nn.Cell):
    def __init__(self, in_c: int, width: int = 256, out_c: int = 256):
        super().__init__()
        
        self.conv1 = nn.SequentialCell(
            nn.Conv1d(in_c, width, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )
        
        self.conv2 = nn.SequentialCell(
            nn.Conv1d(width, width, kernel_size=9, padding=4),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )
        
        self.conv_out = nn.Conv1d(width, out_c, kernel_size=1)
        

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv_out(x)
        
        x = x.permute(0, 2, 1).contiguous()
        
        return x

class Conv2DHead(nn.Cell):
    def __init__(self, in_c: int, width: int = 256, out_c: int = 256, t_embed_dim: int = 512, use_t_embed: bool = True):
        super().__init__()

        if use_t_embed:
            self.norm1 = AdaLayerNormZero(t_embed_dim, in_c)
        else:
            self.norm1 = None

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_c, width, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )

        if self.norm1 is not None:
            self.norm2 = mint.nn.LayerNorm(width, elementwise_affine=False)
        else:
            self.norm2 = None

        self.conv2 = nn.SequentialCell(
            nn.Conv2d(width, width, kernel_size=3, padding=1, pad_mode='pad'),
            nn.GroupNorm(num_groups=min(32, width), num_channels=width),
            nn.SiLU(),
        )
        
        self.conv_out = nn.Conv2d(width, out_c, kernel_size=1)
        

    def construct(self, x: ms.Tensor, t_embed: ms.Tensor) -> ms.Tensor:
        if self.norm1 is not None and t_embed is not None:
            b, c, h, w = x.shape
            residual = x
            x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
            x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x_flat, emb=t_embed)
            x = x_norm.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)

        if self.norm2 is not None:
            x = residual + gate_msa[:, :, None, None] * x
            residual = x
            x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, -1).contiguous()
            x_norm = self.norm2(x_flat)
            x = x_norm * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x = self.conv2(x)

        if self.norm1 is not None and t_embed is not None:
            x = residual + gate_mlp[:, :, None, None] * x

        x = self.conv_out(x)
        
        return x
    

class TransformerHead(nn.Cell):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = mint.nn.LayerNorm(d_model)
        self.attn = ms.nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = mint.nn.LayerNorm(d_model)
        
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(d_model, hidden),
            nn.SiLU(),
            mint.nn.Dropout(dropout),
            mint.nn.Linear(hidden, d_model),
        )
        

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.norm1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        
        x = self.mlp(self.norm2(x))

        return x


class RewardHead(nn.Cell):
    def __init__(
        self,
        token_dim: int,
        conv_type: str = '1d',
        width: int = -1,
        out_dim: int = 1,
        n_visual_heads: int = 1,
        n_text_heads: int = 1,
        patch_size: int = 16,
        t_embed_dim: int = 512,
        use_t_embed: bool = True,
    ):
        super().__init__()
        if conv_type not in ["1d", "2d", "transformer"]:
            conv_type = "1d"
        self.conv_type = conv_type
        
        if width == -1:
            width = token_dim
            
        if conv_type == '1d':
            self.visual_heads = nn.CellList([
                Conv1DHead(token_dim, width, width) for _ in range(n_visual_heads)
            ])
        elif conv_type == '2d':
            self.visual_heads = nn.CellList([
                Conv2DHead(token_dim // (patch_size ** 2), width // (patch_size ** 2), width // (patch_size ** 2), t_embed_dim=t_embed_dim, use_t_embed=use_t_embed) for _ in range(n_visual_heads)
            ])
        elif conv_type == 'transformer':
            self.visual_heads = nn.CellList([
                TransformerHead(token_dim, n_heads=width // 64) for _ in range(n_visual_heads)
            ])
        
        if n_text_heads > 0:
            self.text_heads = nn.CellList([
                TransformerHead(token_dim, n_heads=width // 64) for _ in range(n_text_heads)
            ])
            self.use_text_features = True
        else:
            self.text_heads = None
            
        
        self.out = mint.nn.Linear((n_visual_heads + n_text_heads) * token_dim, out_dim)

        self.patch_size = patch_size  
        
    def unpatchify(self, x, hw=None):
        p = self.patch_size
        c = x.shape[2] // (p ** 2)

        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)).contiguous()
        x = mint.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p)).contiguous()
        return imgs
    
    def patchify(self, imgs):
        p = self.patch_size
        c = imgs.shape[1]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p)).contiguous()
        x = mint.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c)).contiguous()

        return x
    
        
    def construct(
        self,
        visual_features,
        text_features,
        t_embed: Optional[ms.Tensor] = None,
        hw=None) -> ms.Tensor:

        out_features = []
        if self.conv_type in ['1d', 'transformer']:
            for head, visual_feature in zip(self.visual_heads, visual_features):
                visual_feature = head(visual_feature)
                visual_feature = visual_feature.mean(dim=1)
                out_features.append(visual_feature)
                
        elif self.conv_type == '2d':
            for head, visual_feature in zip(self.visual_heads, visual_features):
                visual_feature = self.unpatchify(visual_feature, hw=hw)
                visual_feature = head(visual_feature, t_embed)
                visual_feature = self.patchify(visual_feature)
                visual_feature = visual_feature.mean(dim=1)
                out_features.append(visual_feature)
                
        if self.use_text_features and text_features is not None:
            for head, text_feature in zip(self.text_heads, text_features):
                text_feature = head(text_feature)
                text_feature = text_feature.mean(dim=1)
                out_features.append(text_feature)
                
        out_features = mint.cat(out_features, dim=-1)
        out = self.out(out_features)
        
        return out
        

class RMSNorm(nn.Cell):
    """Torch-like RMSNorm with weight only (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = ms.Parameter(mint.ones((dim,), dtype=ms.float32))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x_f = x.astype(ms.float32)
        var = mint.mean(x_f * x_f, dim=-1, keepdim=True)
        x_norm = x_f * mint.rsqrt(var + self.eps)
        return (x_norm * self.weight.astype(x_norm.dtype)).astype(x.dtype)


class FiLMLayerAdapter(nn.Cell):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_embed = ms.Parameter(mint.randn((1, 1, hidden_dim), dtype=ms.float32) * 0.02)
        self.cond_mlp = nn.SequentialCell(
            mint.nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            mint.nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.proj = mint.nn.Linear(hidden_dim, output_dim)

    def construct(self, x: ms.Tensor, t_emb: ms.Tensor):
        x = x.astype(ms.float32)
        t_emb = t_emb.astype(ms.float32)
        style = self.cond_mlp(t_emb)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        x = x * (1 + gamma) + beta
        x = x + self.layer_embed.astype(x.dtype)
        return self.proj(x)


def _scaled_dot_product_attention(q: ms.Tensor, k: ms.Tensor, v: ms.Tensor) -> ms.Tensor:
    # Single-head attention: (B, Lq, D) x (B, Lk, D) -> (B, Lq, D)
    d = q.shape[-1]
    scores = mint.matmul(q, k.swapaxes(-2, -1)) / math.sqrt(float(d))
    attn = mint.nn.functional.softmax(scores, dim=-1)
    return mint.matmul(attn, v)


class QFormerBlock(nn.Cell):
    def __init__(self, dim: int, use_norm: bool = True, use_text: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim) if use_norm else None
        self.norm2 = RMSNorm(dim) if use_norm else None
        self.norm3 = RMSNorm(dim) if (use_norm and use_text) else None

        self.to_q = mint.nn.Linear(dim, dim)
        self.to_k_vis = mint.nn.Linear(dim, dim)
        self.to_v_vis = mint.nn.Linear(dim, dim)

        self.to_k_text = mint.nn.Linear(dim, dim) if use_text else None
        self.to_v_text = mint.nn.Linear(dim, dim) if use_text else None

        # keep structure to match ckpt keys: to_out.0.{weight,bias}, to_out.1 is dropout (no params)
        self.to_out = nn.CellList([mint.nn.Linear(dim, dim), mint.nn.Dropout(dropout)])

    def construct(self, queries: ms.Tensor, context_visual: ms.Tensor, context_text: Optional[ms.Tensor] = None):
        if self.norm1 is not None:
            queries = self.norm1(queries)
        if self.norm2 is not None:
            context_visual = self.norm2(context_visual)
        if self.norm3 is not None and context_text is not None:
            context_text = self.norm3(context_text)

        q = self.to_q(queries)
        k_vis = self.to_k_vis(context_visual)
        v_vis = self.to_v_vis(context_visual)

        if context_text is not None and self.to_k_text is not None:
            k_text = self.to_k_text(context_text)
            v_text = self.to_v_text(context_text)
            k = mint.cat([k_vis, k_text], dim=1)
            v = mint.cat([v_vis, v_text], dim=1)
        else:
            k = k_vis
            v = v_vis

        hidden_states = _scaled_dot_product_attention(q, k, v)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class RewardHeadV3(nn.Cell):
    """
    QFormer-style reward head (matches Diffusion-RM v3 checkpoints with keys like:
    query_tokens / layer_adapters_visual.* / attn1.* / attn2.* / ff.* / head.*
    """

    def __init__(
        self,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_visual_heads: int = 1,
        n_text_heads: int = 1,
        num_queries: int = 4,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs

        if width == -1:
            width = token_dim

        feature_out_dim = width // 4

        self.layer_adapters_visual = nn.CellList(
            [FiLMLayerAdapter(hidden_dim=width, output_dim=feature_out_dim) for _ in range(n_visual_heads)]
        )
        self.layer_adapters_text = nn.CellList(
            [FiLMLayerAdapter(hidden_dim=width, output_dim=feature_out_dim) for _ in range(n_text_heads)]
        )

        self.agg_visual = mint.nn.Linear(n_visual_heads * feature_out_dim, width) if n_visual_heads > 0 else None
        self.agg_text = mint.nn.Linear(n_text_heads * feature_out_dim, width) if n_text_heads > 0 else None

        self.query_tokens = ms.Parameter(mint.randn((1, num_queries, width), dtype=ms.float32) * 0.02)

        self.attn1 = QFormerBlock(dim=width, dropout=0.0, use_text=(n_text_heads > 0), use_norm=True)
        self.attn2 = QFormerBlock(dim=width, dropout=0.0, use_text=False, use_norm=False)

        self.ff = nn.SequentialCell(
            mint.nn.Linear(width, width * 4),
            nn.GELU(),
            mint.nn.Linear(width * 4, width),
        )

        self.head = mint.nn.Linear(width, out_dim)

    def construct(
        self,
        visual_features,
        text_features,
        t_embed: Optional[ms.Tensor] = None,
        hw=None,
    ) -> ms.Tensor:
        _ = hw
        # Process visual features
        out_visual_features = []
        for adapter, visual_feature in zip(self.layer_adapters_visual, visual_features):
            out_visual_features.append(adapter(visual_feature, t_embed))
        visual_out = None
        if self.agg_visual is not None:
            visual_out = self.agg_visual(mint.cat(out_visual_features, dim=-1))

        # Process text features
        out_text_features = []
        for adapter, text_feature in zip(self.layer_adapters_text, text_features):
            out_text_features.append(adapter(text_feature, t_embed))
        text_out = None
        if self.agg_text is not None:
            text_out = self.agg_text(mint.cat(out_text_features, dim=-1))

        # Prepare query tokens
        bsz = (visual_out if visual_out is not None else text_out).shape[0]
        queries = self.query_tokens.astype((visual_out if visual_out is not None else text_out).dtype).repeat(bsz, 1, 1)

        # Attention blocks
        queries = self.attn1(queries, visual_out, text_out)
        queries = self.attn2(queries, visual_out)
        queries = self.ff(queries)

        scores = self.head(queries).mean(dim=1)
        return scores
