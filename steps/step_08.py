from max.nn import Linear, Module
from max.tensor import Tensor
from step_01 import GPT2Config
from step_07 import MaxGPT2Model


class MaxGPT2LMHeadModel(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = MaxGPT2Model(config=config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B,S]
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
