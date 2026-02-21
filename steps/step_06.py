from max.nn import Module
from max.tensor import Tensor
from step_01 import GPT2Config
from step_02 import GPT2MLP
from step_04 import GPT2MultiHeadAttention
from step_05 import LayerNorm


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
