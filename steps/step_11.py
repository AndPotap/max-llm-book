import torch
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.tensor import TensorType, defaults
from step_01 import GPT2Config
from step_08 import MaxGPT2LMHeadModel
from step_10 import generate_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def run_model() -> None:
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print(f"Loaded HuggingFace model:\n{hf_model}")
    hf_model = hf_model.to(torch.bfloat16)

    state = hf_model.state_dict()
    for key, val in state.items():
        if any(case in key for case in ["c_attn", "c_proj", "c_fc"]):
            state[key] = state[key].T.contiguous()

    _, device = defaults()
    print(f"Using device: {device}")
    config = GPT2Config()
    max_model = MaxGPT2LMHeadModel(config)
    print(f"Model has {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding dim")
    max_model.to(CPU())
    print("On CPU")
    max_model.load_state_dict(state)
    print("Loaded model")
    max_model.to(device)
    print(f"Model on {device=}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nCompiling model...")
    token_type = TensorType(DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device))
    compiled_max_model = max_model.compile(token_type)

    print("\n" + "=" * 50)
    print("Model ready! Enter prompts to generate text.")
    print("Press Ctrl+C or type 'quit' to exit.")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = input("Enter your prompt: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue
            generated_text = generate_text(
                compiled_max_model,
                tokenizer,
                device,
                user_input,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
            )
            print(f"\nGenerated text:\n{generated_text}\n")
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    run_model()
