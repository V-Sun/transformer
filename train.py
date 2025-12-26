"""
Training script for the GPT-style transformer model.

This script trains a character-level language model on text data.
"""

import torch
import torch.nn as nn
from transformer import GPT, GPTConfig, CharDataset


def main():
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available()
                         else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(1337)

    # Load text data
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text):,} characters from input.txt")
    except FileNotFoundError:
        # Use sample text if no file provided
        text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life."""
        print("Using sample text (input.txt not found)")

    # Hyperparameters
    block_size = 128
    batch_size = 32
    num_steps = 5000
    eval_interval = 500
    learning_rate = 3e-4

    # Create dataset
    dataset = CharDataset(text, block_size)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset):,} examples")

    # Initialize model
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        dropout=0.1
    )

    model = GPT(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Training loop
    print(f"\nTraining for {num_steps:,} steps...")
    print("=" * 60)

    model.train()
    for step in range(num_steps):
        # Get batch
        x, y = dataset.get_batch(batch_size)
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % eval_interval == 0 or step == num_steps - 1:
            print(f"Step {step:5d}/{num_steps} | Loss: {loss.item():.4f}")

    print("=" * 60)
    print("Training complete!")

    # Generate sample text
    print("\n" + "=" * 60)
    print("Generating sample text...")
    print("=" * 60)

    model.eval()
    prompt = text[:20] if len(text) >= 20 else text
    context = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    generated_text = dataset.decode(generated[0].tolist())

    print(generated_text)
    print("=" * 60)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
    }, 'model.pt')
    print("\nModel saved to model.pt")


if __name__ == "__main__":
    main()
