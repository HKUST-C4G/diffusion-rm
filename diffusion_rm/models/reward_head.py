from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# FiLM Adapter
# =========================================================
class FiLMLayerAdapter(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden_dim: int, output_dim: int, use_proj_in: bool = True):
        super().__init__()

        if in_dim != hidden_dim or use_proj_in:
            self.proj_in = nn.Linear(in_dim, hidden_dim)
        else:
            self.proj_in = nn.Identity()

        self.layer_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.cond_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

        # start as "no modulation"
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        x: (B, N, C)
        t_emb: (B, C)
        return: (B, N, output_dim)
        """
        x = self.proj_in(x)                       # (B, N, hidden_dim)

        style = self.cond_mlp(t_emb)               # (B, 2C)
        gamma, beta = style.chunk(2, dim=-1)       # each (B, C)
        gamma = gamma.unsqueeze(1)                 # (B, 1, C)
        beta = beta.unsqueeze(1)                   # (B, 1, C)

        x = x * (1 + gamma) + beta
        x = x + self.layer_embed
        return self.proj(x)


# =========================================================
# Cross-Attention Block with token-wise V-gating
# =========================================================
class CrossAttnBlockVGated(nn.Module):
    """
    Cross-attn: queries attend to (visual tokens + optional text tokens)
    Gating: token-wise gate on VALUE only (V <- sigmoid(MLP(ctx_token)) * V)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_norm: bool = True,
        use_text: bool = True,
        dropout: float = 0.0,
        use_v_gating: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_text = use_text
        self.use_v_gating = use_v_gating

        # norms
        self.norm_q = nn.RMSNorm(dim) if use_norm else None
        self.norm_v = nn.RMSNorm(dim) if use_norm else None
        self.norm_t = nn.RMSNorm(dim) if (use_norm and use_text) else None

        # projections
        self.to_q = nn.Linear(dim, dim)

        self.to_k_vis = nn.Linear(dim, dim)
        self.to_v_vis = nn.Linear(dim, dim)

        if use_text:
            self.to_k_text = nn.Linear(dim, dim)
            self.to_v_text = nn.Linear(dim, dim)
        else:
            self.to_k_text = None
            self.to_v_text = None

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # token-wise gates (per token, scalar in last dim)
        # g(ctx_token) in (B, L, 1)
        if self.use_v_gating:
            self.gate_vis = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, 1),
            )
            nn.init.zeros_(self.gate_vis[-1].weight)
            nn.init.zeros_(self.gate_vis[-1].bias)

            if self.use_text:
                self.gate_text = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, 1),
                )
                nn.init.zeros_(self.gate_text[-1].weight)
                nn.init.zeros_(self.gate_text[-1].bias)
            else:
                self.gate_text = None
        else:
            self.gate_vis = None
            self.gate_text = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Hd)
        b, l, d = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _unshape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Hd) -> (B, L, D)
        b, h, l, hd = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * hd)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: (B, L, D)
        qh = self._shape(q)
        kh = self._shape(k)
        vh = self._shape(v)
        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            dropout_p=0.0,
            is_causal=False
        )
        return self._unshape(out)

    def forward(
        self,
        queries: torch.Tensor,            # (B, Nq, D)
        context_visual: torch.Tensor,     # (B, Nv, D)
        context_text: Optional[torch.Tensor] = None,  # (B, Nt, D) or None
    ):
        residual = queries

        if self.norm_q is not None:
            queries = self.norm_q(queries)
        if self.norm_v is not None:
            context_visual = self.norm_v(context_visual)
        if self.norm_t is not None and context_text is not None:
            context_text = self.norm_t(context_text)

        q = self.to_q(queries)

        # -------- visual branch --------
        k_vis = self.to_k_vis(context_visual)
        v_vis = self.to_v_vis(context_visual)

        if self.use_v_gating and self.gate_vis is not None:
            g_vis = torch.sigmoid(self.gate_vis(context_visual))  # (B, Nv, 1)
            v_vis = v_vis * g_vis

        h_vis = self._sdpa(q, k_vis, v_vis)

        # -------- text branch (optional) --------
        if self.use_text and (context_text is not None) and (self.to_k_text is not None):
            k_text = self.to_k_text(context_text)
            v_text = self.to_v_text(context_text)

            if self.use_v_gating and (self.gate_text is not None):
                g_text = torch.sigmoid(self.gate_text(context_text))  # (B, Nt, 1)
                v_text = v_text * g_text

            h_text = self._sdpa(q, k_text, v_text)
            h = h_vis + h_text
        else:
            h = h_vis

        out = self.to_out(h)
        return residual + out


# =========================================================
# Reward Head: FiLM adapters + 2x attn + FFN + mean pooling
# =========================================================
class RewardHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_visual_heads: int = 1,
        n_text_heads: int = 1,
        num_queries: int = 4,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        t_embed_dim: int = -1,
        use_proj_in: bool = False,
        **kwargs
    ):
        super().__init__()
        if width == -1:
            width = token_dim

        feature_out_dim = width // 4

        self.layer_adapters_visual = nn.ModuleList([
            FiLMLayerAdapter(in_dim=token_dim, emb_dim=t_embed_dim if t_embed_dim > 0 else width, hidden_dim=width, output_dim=feature_out_dim, use_proj_in=use_proj_in)
            for _ in range(n_visual_heads)
        ])
        self.layer_adapters_text = nn.ModuleList([
            FiLMLayerAdapter(in_dim=token_dim, emb_dim=t_embed_dim if t_embed_dim > 0 else width, hidden_dim=width, output_dim=feature_out_dim, use_proj_in=use_proj_in)
            for _ in range(n_text_heads)
        ])

        self.agg_visual = nn.Linear(n_visual_heads * feature_out_dim, width) if n_visual_heads > 0 else None
        self.agg_text = nn.Linear(n_text_heads * feature_out_dim, width) if n_text_heads > 0 else None

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, width) * 0.02)

        # attn1: vis + text, token-wise V-gate
        self.attn1 = CrossAttnBlockVGated(
            dim=width,
            num_heads=num_attn_heads,
            use_norm=True,
            use_text=(n_text_heads > 0),
            dropout=dropout,
            use_v_gating=True,
        )
        # attn2: visual-only refine (no text)
        self.attn2 = CrossAttnBlockVGated(
            dim=width,
            num_heads=num_attn_heads,
            use_norm=True,
            use_text=False,
            dropout=dropout,
            use_v_gating=False,   # keep it simple
        )

        # FFN residual
        self.norm_ff = nn.RMSNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(width, out_dim)

    def _build_view_tokens(
        self,
        visual_features: List[torch.Tensor],
        text_features: Optional[List[torch.Tensor]],
        t_emb: torch.Tensor,
    ):
        """
        visual_features: list of (B, Nv, C)
        text_features: list of (B, Nt, C) or None
        t_emb: (B, C)

        return: visual_out (B, Nv, width), text_out (B, Nt, width) or None
        """
        # visual
        assert len(visual_features) == len(self.layer_adapters_visual), (
            f"visual_features has {len(visual_features)} tensors, "
            f"but n_visual_heads={len(self.layer_adapters_visual)}"
        )
        out_v = []
        for adapter, vf in zip(self.layer_adapters_visual, visual_features):
            out_v.append(adapter(vf, t_emb))
        out_v = torch.cat(out_v, dim=-1)
        visual_out = self.agg_visual(out_v)

        # text
        if (text_features is None) or (len(self.layer_adapters_text) == 0):
            return visual_out, None

        assert len(text_features) == len(self.layer_adapters_text), (
            f"text_features has {len(text_features)} tensors, "
            f"but n_text_heads={len(self.layer_adapters_text)}"
        )
        out_t = []
        for adapter, tf in zip(self.layer_adapters_text, text_features):
            out_t.append(adapter(tf, t_emb))
        out_t = torch.cat(out_t, dim=-1)
        text_out = self.agg_text(out_t)

        return visual_out, text_out

    def forward(
        self,
        visual_features: List[torch.Tensor],
        text_features: Optional[List[torch.Tensor]],
        t_embed: Optional[torch.Tensor] = None,
        hw=None,
    ):
        """
        visual_features: list of (B, Nv, C)
        text_features:   list of (B, Nt, C) or None
        t_embed: (B, C)
        """
        assert t_embed is not None, "t_embed is required (FiLM adapters depend on it)."

        visual_out, text_out = self._build_view_tokens(visual_features, text_features, t_embed)

        B = visual_out.size(0)
        queries = self.query_tokens.expand(B, -1, -1)

        # attn1: vis+text with token-wise V-gate
        queries = self.attn1(queries, visual_out, text_out)
        # attn2: visual-only refine
        queries = self.attn2(queries, visual_out, None)

        # FFN
        queries = queries + self.ff(self.norm_ff(queries))

        per_query = self.head(queries)  # (B, Q, out_dim)

        # simplest aggregation: mean over queries
        score = per_query.mean(dim=1)   # (B, out_dim)
        return score

    def forward_ensemble(
        self,
        visual_features_per_t: List[List[torch.Tensor]],
        text_features_per_t: Optional[List[List[torch.Tensor]]],
        t_embed_per_t: torch.Tensor,  # (B, K, C)
    ):
        """
        Multi-timestep ensemble: concat tokens across timesteps, then run same reward head once.

        visual_features_per_t: length K list, each item is list[n_visual_heads] of (B, Nv, C)
        text_features_per_t:   length K list, each item is list[n_text_heads] of (B, Nt, C) or None
        t_embed_per_t: (B, K, C)
        """
        assert t_embed_per_t.dim() == 3, "t_embed_per_t must be (B,K,C)"
        B, K, C = t_embed_per_t.shape
        device = t_embed_per_t.device

        visual_all = []
        text_all = []

        for i in range(K):
            t_emb = t_embed_per_t[:, i, :]
            
            vf_i = visual_features_per_t[i]
            tf_i = text_features_per_t[i] if text_features_per_t is not None else None

            v_out_i, t_out_i = self._build_view_tokens(vf_i, tf_i, t_emb)
            visual_all.append(v_out_i)
            if t_out_i is not None:
                text_all.append(t_out_i)

        visual_cat = torch.cat(visual_all, dim=1)  # (B, sum Nv, width)
        text_cat = torch.cat(text_all, dim=1) if len(text_all) > 0 else None

        queries = self.query_tokens.expand(B, -1, -1)

        queries = self.attn1(queries, visual_cat, text_cat)
        queries = self.attn2(queries, visual_cat, None)
        queries = queries + self.ff(self.norm_ff(queries))

        per_query = self.head(queries)
        score = per_query.mean(dim=1)
        return score
