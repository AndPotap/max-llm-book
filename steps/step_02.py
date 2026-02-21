import max.functional as F
from max.nn import Linear
from max.nn.module import Module
from max.tensor import Tensor
from step_01 import GPT2Config


class GPT2MLP(Module):
    def __init__(self, intermediate_size: int, config: GPT2Config) -> None:
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Linear(in_dim=embed_dim, out_dim=intermediate_size, bias=True)
        self.c_proj = Linear(in_dim=intermediate_size, out_dim=embed_dim, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
