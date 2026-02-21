from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int = 50_257
    n_positions: int = 1_024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3_072
    layer_norm_epsilon: float = 1e-5
