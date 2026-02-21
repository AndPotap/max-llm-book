from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Linear
from max.tensor import TensorType, defaults
from step_01 import GPT2Config
from step_08 import MaxGPT2LMHeadModel
from step_10 import generate_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def run_model() -> None:
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print(f"Loaded HuggingFace model:\n{hf_model}")

    _, device = defaults()
    print(f"Using device: {device}")
    config = GPT2Config()
    max_model = MaxGPT2LMHeadModel(config)
    print(f"Model has {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding dim")

    max_model.load_state_dict(hf_model.state_dict())
    max_model.to(device)

    for name, child in max_model.descendents:
        if isinstance(child, Linear):
            if any(layer_name in name for layer_name in ["c_attn", "c_proj", "c_fc"]):
                print(f"Transposing {name}: {child.weight.shape}")
                child.weight = child.weight.T

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
