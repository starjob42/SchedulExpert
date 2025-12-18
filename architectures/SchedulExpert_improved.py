# Path: architectures/SchedulExpert_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, ModuleList, MultiheadAttention
from torch_geometric.nn import GATv2Conv, JumpingKnowledge


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _degree_pe(num_nodes: int, edge_index: torch.Tensor, eps: float = 1e-6):
    """
    Compute simple directed degree positional encodings:
      - deg_in, deg_out, deg_total (all normalized by max-degree)
    Returns: (num_nodes, 3)
    """
    src, dst = edge_index
    deg_out = torch.bincount(src, minlength=num_nodes).float()
    deg_in = torch.bincount(dst, minlength=num_nodes).float()
    deg_tot = deg_in + deg_out
    maxv = deg_tot.max().clamp_min(eps)
    pe = torch.stack([deg_in / maxv, deg_out / maxv, deg_tot / maxv], dim=-1)
    return pe


def _act_fn(name: str):
    name = (name or "relu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky":
        return nn.LeakyReLU(0.15)
    return nn.ReLU()


# -----------------------------------------------------------------------------
# Residual GAT Block (pre-LN, projection residual)
# -----------------------------------------------------------------------------
class ResGATBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_per_head: int,
                 n_heads: int,
                 attn_dropout: float = 0.1,
                 mlp_dropout: float = 0.1,
                 leaky: float = 0.15,
                 act: str = "relu"):
        super().__init__()
        out_dim = hidden_per_head * n_heads
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.ln = LayerNorm(in_dim)
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_per_head,
            heads=n_heads,
            concat=True,                # outputs (N, hidden_per_head * n_heads)
            add_self_loops=False,
            negative_slope=leaky,
            dropout=attn_dropout
        )
        self.drop = Dropout(mlp_dropout)
        self.proj = (nn.Identity() if in_dim == out_dim else Linear(in_dim, out_dim))
        self.act = _act_fn(act)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.ln(x)
        h = self.gat(h, edge_index)
        h = self.drop(h)
        # Simple post-activation helps stability a bit for deeper stacks
        h = self.act(h)
        return self.proj(x) + h


# -----------------------------------------------------------------------------
# Sparse Top-2 MoE
# -----------------------------------------------------------------------------
class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = None, p: float = 0.0, act: str = "relu"):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            Linear(dim, hidden),
            _act_fn(act),
            Dropout(p),
            Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


class Top2Router(nn.Module):
    """
    Switch-style Top-2 router with capacity factor and a simple load-balancing
    auxiliary term (to be added to loss by the trainer).
    """
    def __init__(self, dim: int, n_experts: int, capacity_factor: float = 1.25):
        super().__init__()
        self.w_g = Linear(dim, n_experts)
        nn.init.xavier_uniform_(self.w_g.weight)
        nn.init.zeros_(self.w_g.bias)
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor):
        # x: (T, D)
        logits = self.w_g(x)                 # (T, E)
        gates = torch.softmax(logits, dim=-1)
        topw, topi = torch.topk(gates, k=2, dim=-1)  # (T, 2)
        mean_usage = gates.mean(dim=0)       # (E,)
        return topi, topw, mean_usage


def moe_top2_apply(x: torch.Tensor,
                   experts: nn.ModuleList,
                   router: Top2Router):
    """
    Dispatch tokens to experts with Top-2 gating; enforce capacity.
    Returns: y (T, D), aux_loss (scalar)
    """
    T, D = x.shape
    E = len(experts)
    topi, topw, mean_usage = router(x)      # (T,2),(T,2),(E,)
    capacity = int((T * router.capacity_factor) / E) + 1

    y = torch.zeros_like(x)
    loads = x.new_zeros(E)

    for e in range(E):
        m1 = (topi[:, 0] == e)
        m2 = (topi[:, 1] == e)
        tokens = torch.nonzero(m1 | m2, as_tuple=False).squeeze(-1)
        if tokens.numel() == 0:
            continue
        tokens = tokens[:capacity]  # truncate to capacity
        x_e = x[tokens]
        out_e = experts[e](x_e)     # (Te, D)
        # weight choice: pick the right top weight per token
        w_e = torch.where(m1[tokens], topw[tokens, 0], topw[tokens, 1]).unsqueeze(-1)
        y[tokens] += w_e * out_e
        loads[e] = tokens.numel()

    # aux: encourage balanced expert usage vs. observed load
    if loads.sum() > 0:
        obs = loads / loads.sum()
        aux = (mean_usage * obs).sum()
    else:
        aux = x.new_tensor(0.0)
    return y, aux


# -----------------------------------------------------------------------------
# Encoder: deep residual GAT + JK + optional degree PEs + optional MoE
#   - same signature: forward(x, edge_index) -> (N, input_size + hidden_size)
# -----------------------------------------------------------------------------
class GATEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 # internal width per head; final out is still input_size + hidden_size
                 n_heads: int = 4,
                 n_layers: int = 3,
                 enc_jk: str = "cat",              # none|cat|max|lstm
                 use_degree_pe: bool = True,
                 enc_act: str = "relu",
                 # bottleneck after JK aggregation (before MoE)
                 enc_embed: int = 128,
                 # MoE
                 use_moe: bool = False,
                 n_experts: int = 4,
                 expert_hidden: int = None,
                 expert_dropout: float = 0.0,
                 moe_capacity: float = 1.25,
                 attn_dropout: float = 0.1,
                 mlp_dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_degree_pe = use_degree_pe
        self.enc_jk = enc_jk.lower()
        self.use_moe = use_moe

        pe_extra = 3 if use_degree_pe else 0
        stem_in = input_size + pe_extra
        # First project into the block width = hidden_per_head * n_heads
        block_width = hidden_size * n_heads
        self.stem = Linear(stem_in, block_width)

        # Stack of residual GAT blocks with constant width
        blocks = []
        in_dim = block_width
        for _ in range(n_layers):
            blocks.append(
                ResGATBlock(
                    in_dim=in_dim,
                    hidden_per_head=hidden_size,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                    act=enc_act
                )
            )
            in_dim = block_width  # remains constant
        self.blocks = ModuleList(blocks)

        # Jumping Knowledge (optional)
        if self.enc_jk == "none":
            self.jk = None
            jk_dim = block_width
        else:
            self.jk = JumpingKnowledge(mode=self.enc_jk)
            # for 'cat', output dim is L * block_width; for 'max'/'lstm', it is block_width
            jk_dim = (block_width * n_layers) if self.enc_jk == "cat" else block_width

        # Fuse to embedding size (pre-MoE)
        self.fuse = Linear(jk_dim, enc_embed)
        self.fuse_ln = LayerNorm(enc_embed)
        self.fuse_act = _act_fn(enc_act)

        # MoE (optional) on enc_embed
        if use_moe:
            self.experts = ModuleList([
                ExpertMLP(enc_embed, hidden=expert_hidden or enc_embed,
                          p=expert_dropout, act=enc_act)
                for _ in range(n_experts)
            ])
            self.router = Top2Router(enc_embed, n_experts=n_experts, capacity_factor=moe_capacity)
        else:
            self.experts = None
            self.router = None

        # Final projection to hidden_size (to keep out_size = input_size + hidden_size)
        self.final_proj = Linear(enc_embed, hidden_size)
        self.out_size = input_size + hidden_size

        # will be populated during forward if MoE is used
        self._aux = None

    @property
    def aux_loss(self):
        return self._aux if (self._aux is not None) else 0.0

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x: (N, input_size)
        edge_index: (2, E)
        returns: (N, input_size + hidden_size)
        """
        N = x.size(0)

        if self.use_degree_pe:
            pe = _degree_pe(N, edge_index).to(x.dtype).to(x.device)
            x_in = torch.cat([x, pe], dim=-1)
        else:
            x_in = x

        h = self.stem(x_in)  # (N, block_width)

        feats = []
        for blk in self.blocks:
            h = blk(h, edge_index)  # (N, block_width)
            if self.jk is not None:
                feats.append(h)

        if self.jk is None:
            agg = h
        else:
            if self.enc_jk == "cat":
                # JumpingKnowledge expects List[Tensor]
                agg = self.jk(feats)
            else:
                agg = self.jk(feats)

        fused = self.fuse_act(self.fuse_ln(self.fuse(agg)))  # (N, enc_embed)

        if self.use_moe:
            moe_out, aux = moe_top2_apply(fused, self.experts, self.router)
            self._aux = aux
            fused = moe_out
        else:
            self._aux = None

        out_h = self.fuse_act(self.final_proj(fused))        # (N, hidden_size)
        return torch.cat([x, out_h], dim=-1)                 # (N, input_size + hidden_size)


# -----------------------------------------------------------------------------
# FiLM conditioning + optional global context cross-attention
# -----------------------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.gamma = Linear(cond_dim, target_dim)
        self.beta = Linear(cond_dim, target_dim)
        self.ln = LayerNorm(target_dim)

    def forward(self, target: torch.Tensor, cond: torch.Tensor):
        # both (B, N, D)
        g = self.gamma(cond)
        b = self.beta(cond)
        return self.ln(g * target + b)


class MHADecoder(nn.Module):
    """
    Decoder: memory net over per-job state + FiLM modulation of candidate embeddings,
    with optional global context cross-attention.
    Signature unchanged vs original:
      forward(embed_x: (B,N,encoder_size), state: (B,N,context_size)) -> logits (B,N)
    """
    def __init__(self,
                 encoder_size: int,
                 context_size: int,
                 hidden_size: int = 64,
                 mem_size: int = 128,
                 clf_size: int = 128,
                 n_heads: int = 4,
                 attn_dropout: float = 0.1,
                 mlp_dropout: float = 0.1,
                 leaky_slope: float = 0.15,
                 use_film: bool = True,
                 use_dec_global_attn: bool = False):
        super().__init__()
        # Memory net
        self.linear1 = Linear(context_size, hidden_size * n_heads)
        self.ln1 = LayerNorm(hidden_size * n_heads)
        self.self_attn = MultiheadAttention(hidden_size * n_heads, num_heads=n_heads,
                                            dropout=attn_dropout, batch_first=True)
        self.ln2 = LayerNorm(hidden_size * n_heads)
        self.linear2 = Linear(hidden_size * n_heads, mem_size)

        # FiLM
        self.use_film = use_film
        if use_film:
            self.film = FiLM(context_size, encoder_size)

        # Optional global context attention (project to a shared attn dim)
        self.use_dec_global_attn = use_dec_global_attn
        if use_dec_global_attn:
            attn_dim = hidden_size * n_heads
            self.q_proj = Linear(encoder_size, attn_dim)
            self.kv_proj = Linear(encoder_size, attn_dim)
            self.global_attn = MultiheadAttention(attn_dim, num_heads=n_heads,
                                                  dropout=attn_dropout, batch_first=True)
            self.global_ln = LayerNorm(attn_dim)
            self.back_proj = Linear(attn_dim, encoder_size)

        # Classifier head
        self.act = nn.LeakyReLU(leaky_slope)
        self.linear3 = Linear(encoder_size + mem_size, clf_size)
        self.ln3 = LayerNorm(clf_size)
        self.drop = Dropout(mlp_dropout)
        self.linear4 = Linear(clf_size, 1)

    def forward(self, embed_x: torch.Tensor, state: torch.Tensor):
        """
        embed_x: (B, N, encoder_size)
        state:   (B, N, context_size)
        returns logits: (B, N)
        """
        B, N, D = embed_x.shape

        # Memory over states
        x1 = self.ln1(self.linear1(state))
        x2 = x1 + self.self_attn(x1, x1, x1)[0]
        x2 = self.ln2(x2)
        x2 = torch.relu(self.linear2(x2))  # (B, N, mem_size)

        # FiLM modulation
        mod = self.film(embed_x, state) if self.use_film else embed_x  # (B, N, D)

        # Global context cross-attention (optional)
        if self.use_dec_global_attn:
            g = mod.mean(dim=1, keepdim=True)                  # (B,1,D)
            q = self.q_proj(mod)                               # (B,N,A)
            k = self.kv_proj(g)                                # (B,1,A)
            v = k                                              # (B,1,A)
            attn_out, _ = self.global_attn(q, k, v)            # (B,N,A)
            attn_out = self.global_ln(attn_out)
            mod = mod + self.back_proj(attn_out)               # (B,N,D)

        # Classifier
        z = torch.cat([mod, x2], dim=-1)                       # (B,N,D+mem)
        z = self.act(self.linear3(z))
        z = self.ln3(self.drop(z))
        return self.linear4(z).squeeze(-1)                     # (B,N)