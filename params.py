from transformers import AutoModelForCausalLM
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--model_path", type=str, default="./models/GPT2-Large-BabyLM",
                        help="Path to the model directory")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype="auto")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params

    print("\n===== Parameter Count =====")
    print(f"Total parameters       : {total_params:,}")
    print(f"Trainable parameters   : {trainable_params:,}")
    print(f"Non-trainable parameters: {nontrainable_params:,}")

    # Estimated size
    bytes_total = total_params * 4   # float32 assumed
    print(f"\nApprox total size (float32): {bytes_total / (1024**3):.2f} GB")

    # official GPT-2 Large number
    official_gpt2_large_params = 774_030_080
    print(f"\nGPT-2 Large official parameter count: {official_gpt2_large_params:,}")
    print(f"Difference: {total_params - official_gpt2_large_params:,}")

if __name__ == "__main__":
    main()
