"""
Text generation script using a trained GPT model.

Usage:
    python generate.py --prompt "Your prompt here" --max_tokens 100
"""

import argparse
import torch
from transformer import GPT


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained GPT model")
    parser.add_argument("--model", type=str, default="model.pt", help="Path to saved model")
    parser.add_argument("--prompt", type=str, default="To be", help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available()
                         else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)

    model = GPT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {checkpoint['vocab_size']}")

    # Encode prompt
    def encode(text):
        return [stoi[ch] for ch in text if ch in stoi]

    def decode(indices):
        return ''.join([itos[i] for i in indices])

    # Generate text
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

    generated_text = decode(generated[0].tolist())
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
