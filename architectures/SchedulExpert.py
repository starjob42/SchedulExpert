import torch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU, MultiheadAttention, LayerNorm

class ExpertModule(torch.nn.Module):
    """
    A simple expert module consisting of a two-layer MLP.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super(ExpertModule, self).__init__()
        self.network = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


class AttentionRouter(torch.nn.Module):
    """
    Attention-based router to dynamically assign weights to different experts.
    """
    def __init__(self, embed_dim: int, n_experts: int, hidden_dim: int = 64, n_heads: int = 8, dropout: float = 0.5):
        super(AttentionRouter, self).__init__()
        self.query = Linear(embed_dim, hidden_dim)
        self.key = Linear(embed_dim, hidden_dim)
        self.value = Linear(embed_dim, hidden_dim)
        self.attn = MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = LayerNorm(hidden_dim)
        self.layer_norm2 = LayerNorm(hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.n_experts = n_experts
        self.experts = ModuleList([
            ExpertModule(input_dim=hidden_dim, output_dim=hidden_dim) for _ in range(n_experts)
        ])
        
        # Learnable layer to compute expert scores from attention output
        self.expert_selector = Linear(hidden_dim, n_experts)
        torch.nn.init.xavier_uniform_(self.expert_selector.weight)
        if self.expert_selector.bias is not None:
            torch.nn.init.zeros_(self.expert_selector.bias)
        
        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, embed: torch.Tensor):
        """
        Args:
            embed: Tensor of shape (batch_size, embed_dim=128)
        Returns:
            routed_embed: Tensor of shape (batch_size, hidden_dim=64)
            expert_weights: Tensor of shape (batch_size, n_experts)
        """
        # Add a sequence dimension
        Q = self.query(embed).unsqueeze(1)  # (batch_size, 1, hidden_dim=64)
        K = self.key(embed).unsqueeze(1)   # (batch_size, 1, hidden_dim=64)
        V = self.value(embed).unsqueeze(1) # (batch_size, 1, hidden_dim=64)

        attn_output, attn_weights = self.attn(Q, K, V)  # (batch_size, 1, hidden_dim=64)

        # Apply LayerNorm after attention
        attn_output = self.layer_norm2(attn_output)

        # Mean over the sequence dim (which is 1)
        attn_summary = attn_output.mean(dim=1)  # (batch_size, hidden_dim=64)

        # Compute expert scores
        expert_scores = self.expert_selector(attn_summary)  # (batch_size, n_experts)

        # Apply softmax and dropout to obtain expert weights
        expert_weights = self.dropout(self.softmax(expert_scores))  # (batch_size, n_experts)

        # Process each expert
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(attn_output.squeeze(1))  # (batch_size, hidden_dim=64)
            expert_outputs.append(expert_out.unsqueeze(1))  # (batch_size, 1, hidden_dim=64)

        # Stack expert outputs -> (batch_size, n_experts, hidden_dim=64)
        expert_outputs = torch.cat(expert_outputs, dim=1)

        # Weighted sum of expert outputs
        expert_weights = expert_weights.unsqueeze(-1)  # (batch_size, n_experts, 1)
        routed_embed = torch.sum(expert_weights * expert_outputs, dim=1)  # (batch_size, hidden_dim=64)

        return routed_embed, expert_weights


class GATEncoder(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 embed_size: int = 128,  # Must be divisible by n_experts
                 n_heads: int = 3,
                 leaky_slope: float = 0.15,
                 n_experts: int = 8):
        """
        Args:
            input_size (int): Node feature dimension.
            hidden_size (int): GAT hidden size.
            embed_size (int): GAT final embedding size (before experts).
            n_heads (int): Number of GAT heads.
            n_experts (int): Number of experts for mixture-of-experts.
        """
        super(GATEncoder, self).__init__()
        self.embedding1 = GATv2Conv(in_channels=input_size,
                                    out_channels=hidden_size,
                                    dropout=0.5,
                                    heads=n_heads,
                                    concat=True,
                                    add_self_loops=False,
                                    negative_slope=leaky_slope)
        self.embedding2 = GATv2Conv(in_channels=hidden_size * n_heads + input_size,
                                    out_channels=embed_size,
                                    dropout=0.5,
                                    heads=n_heads,
                                    concat=False,
                                    add_self_loops=False,
                                    negative_slope=leaky_slope)

        # The final output of the encoder is (x + routed_embed)
        # So the out_size = input_size + (the final MoE dimension)
        self.out_size = input_size + hidden_size  # e.g., 15 + 64 = 79

        # Mixture of Experts logic
        assert embed_size % n_experts == 0, (
            f"embed_size ({embed_size}) must be divisible by n_experts ({n_experts})."
        )
        self.n_experts = n_experts
        self.partition_size = embed_size // n_experts
        self.experts = ModuleList([
            ExpertModule(input_dim=self.partition_size, output_dim=self.partition_size)
            for _ in range(n_experts)
        ])

        # Dynamic routing using AttentionRouter
        self.router = AttentionRouter(embed_dim=embed_size, n_experts=n_experts)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            x: Node features of shape (num_nodes, input_size)
            edge_index: Graph edges
        Returns:
            A tensor of shape (num_nodes, input_size + hidden_size)
        """
        h1 = torch.relu(self.embedding1(x, edge_index))
        h = torch.cat([x, h1], dim=-1)
        h2 = torch.relu(self.embedding2(h, edge_index))
        # h2: (num_nodes, embed_size)

        # Partition the embedding among experts
        partitions = torch.chunk(h2, self.n_experts, dim=-1)  # each shape (num_nodes, partition_size)
        specialized = [expert(part) for expert, part in zip(self.experts, partitions)]
        specialized_embed = torch.cat(specialized, dim=-1)  # (num_nodes, embed_size)

        # Dynamic routing
        routed_embed, expert_weights = self.router(specialized_embed)  # (num_nodes, 64)

        # Concatenate the original x to maintain the final dimension (79, for example)
        return torch.cat([x, routed_embed], dim=-1)  # (num_nodes, input_size + hidden_size)


class MHADecoder(torch.nn.Module):
    """
    Simple Decoder that uses self-attention (MultiheadAttention) to produce logits over jobs.
    """
    def __init__(self,
                 encoder_size: int,
                 context_size: int,
                 hidden_size: int = 64,
                 mem_size: int = 128,
                 clf_size: int = 128,
                 leaky_slope: float = 0.15,
                 n_heads: int = 3):
        super(MHADecoder, self).__init__()
        # Memory net
        self.linear1 = torch.nn.Linear(context_size, hidden_size * n_heads)
        self.layer_norm1 = LayerNorm(hidden_size * n_heads)
        self.linear2 = torch.nn.Linear(hidden_size * n_heads, mem_size)
        self.self_attn = torch.nn.MultiheadAttention(
            hidden_size * n_heads,
            num_heads=n_heads,
            dropout=0.5,
            batch_first=True
        )
        self.layer_norm2 = LayerNorm(hidden_size * n_heads)
        
        # Classifier net
        self.act = torch.nn.LeakyReLU(leaky_slope)
        self.linear3 = torch.nn.Linear(encoder_size + mem_size, clf_size)
        self.layer_norm3 = LayerNorm(clf_size)
        self.linear4 = torch.nn.Linear(clf_size, 1)

    def forward(self, embed_x: torch.Tensor, state: torch.Tensor):
        """
        Args:
            embed_x: (B, N, encoder_size)
            state:   (B, N, context_size)
        Returns:
            logits: (B, N)
        """
        # Memory Network
        x1 = self.linear1(state)  # (B, N, hidden_size * n_heads)
        x1 = self.layer_norm1(x1)
        x2 = x1 + self.self_attn(x1, x1, x1)[0]  # (B, N, hidden_size * n_heads)
        x2 = self.layer_norm2(x2)
        x2 = torch.relu(self.linear2(x2))  # (B, N, mem_size)

        # Classifier Network
        combined = torch.cat([embed_x, x2], dim=-1)  # (B, N, encoder_size + mem_size)
        combined = self.act(self.linear3(combined))  # (B, N, clf_size)
        combined = self.layer_norm3(combined)
        return self.linear4(combined).squeeze(-1)  # (B, N)
