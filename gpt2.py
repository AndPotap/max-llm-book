import math
from dataclasses import dataclass

import max.functional as F
import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import Dim, DimLike
from max.nn import Embedding, Linear, Module, Sequential
from max.tensor import Tensor
from transformers import GPT2Tokenizer


def encode_text(text: str, tokenizer: GPT2Tokenizer, device: Device, max_length: int) -> Tensor:
    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    return Tensor.constant([token_ids], dtype=DType.int64, device=device)


def decode_tokens(token_ids: Tensor, tokenizer: GPT2Tokenizer) -> str:
    token_ids_np = np.from_dlpack(token_ids.to(CPU()))

    if token_ids_np.ndim > 1:
        token_ids_np = token_ids_np.flatten()

    token_ids_list: list = token_ids_np.tolist()
    return tokenizer.decode(token_ids_list, skip_special_tokens=True)


def generate_text(
    model: Module,
    tokenizer: GPT2Tokenizer,
    device: Device,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> str:
    generated_tokens = encode_text(prompt, tokenizer, device, max_length=100)

    print(f"Starting generation from: '{prompt}'")
    print(f"Settings: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}")
    print("-" * 50)

    for step in range(max_new_tokens):
        logits = model(generated_tokens)
        next_token_logits = logits[0, -1, :]

        if do_sample and temperature > 0:
            temp_tensor = Tensor.constant(
                temperature,
                dtype=next_token_logits.dtype,
                device=next_token_logits.device,
            )
            next_token_logits = next_token_logits / temp_tensor
            probs = F.softmax(next_token_logits)

            probs_np: np.ndarray = np.from_dlpack(probs.to(CPU()))
            if probs_np.ndim > 1:
                probs_np = probs_np.flatten()
            probs_np = probs_np.astype(np.float64)
            next_token_id = np.random.choice(len(probs_np), p=probs_np)
            next_token_tensor = Tensor.constant(next_token_id, dtype=DType.int64, device=generated_tokens.device)
        else:
            next_token_tensor = F.argmax(next_token_logits)

        next_token_2d = next_token_tensor.reshape([1, -1])
        generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)

        if step % 5 == 0 or step == max_new_tokens - 1:
            current_text = decode_tokens(generated_tokens, tokenizer)
            print(f"Step {step + 1:2d}: {current_text}")

    final_text = decode_tokens(generated_tokens, tokenizer)
    print("-" * 50)
    print(f"Final generated text: '{final_text}'")
    return final_text


@dataclass
class GPT2Config:
    vocab_size: int = 50_257
    n_positions: int = 1_024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3_072
    layer_norm_epsilon: float = 1e-5


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


@F.functional
def causal_mask(seq_length: DimLike, num_tokens: DimLike, *, dtype: DType, device: Device) -> Tensor:
    n = Dim(seq_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(seq_length, n))
    mask = F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
    return mask


class GPT2MultiHeadAttention(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor: Tensor, num_heads: int, attn_head_dim: int) -> Tensor:
        new_shape = list(tensor.shape[:-1]) + [num_heads, attn_head_dim]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)

    def _merge_heads(self, tensor: Tensor, num_heads: int, attn_head_dim: int) -> Tensor:
        tensor = tensor.transpose(-3, -2)
        new_shape = list(tensor.shape[:-2]) + [num_heads * attn_head_dim]
        return tensor.reshape(new_shape)

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        _, num_heads, seq_length, head_dim = query.shape
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(int(head_dim))
        mask = causal_mask(seq_length, 0, dtype=query.dtype, device=query.device)
        attn_weights = F.softmax(attn_weights + mask)
        output = attn_weights @ value
        return output

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: [B,S,D]
        qkv = self.c_attn(hidden_states)
        query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)
        return attn_output


class LayerNorm(Module):
    def __init__(self, dim: DimLike, *, eps: float) -> None:
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
