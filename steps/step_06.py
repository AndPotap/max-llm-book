import max.functional as F
from max.dtype import DType
from max.graph import DimLike
from max.nn import Module
from max.tensor import Tensor
from step_01 import GPT2Config
from step_02 import GPT2MLP
from step_04 import GPT2MultiHeadAttention


class LayerNorm(Module):
    def __init__(self, dim: DimLike, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Tensor.ones(shape=[dim], dtype=DType.bfloat16)
        self.bias = Tensor.zeros(shape=[dim], dtype=DType.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        # x: [...,D]
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


class GPT2Block(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        hidden_size = config.n_embd
        inner_dim = config.n_inner if hasattr(config, "n_inner") and config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNorm(dim=hidden_size, eps=config.layer_norm_epsilon)

        self.attn = GPT2MultiHeadAttention(config=config)

        self.ln_2 = LayerNorm(dim=hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(intermediate_size=inner_dim, config=config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states
