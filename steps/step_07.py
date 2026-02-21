from max.nn import Embedding, Module, Sequential
from max.tensor import Tensor
from step_01 import GPT2Config
from step_06 import GPT2Block, LayerNorm


class MaxGPT2Model(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_length = input_ids.shape

        tok_embeds = self.wte(input_ids)
        position_indices = Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        pos_embeds = self.wpe(position_indices)

        x = tok_embeds + pos_embeds
        x = self.h(x)
        x = self.ln_f(x)
        return x
