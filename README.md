# GPT Transformer from Scratch

A clean, educational implementation of a GPT-style decoder-only transformer built from scratch using PyTorch. This project demonstrates the core components of modern language models including multi-head attention, positional embeddings, and autoregressive text generation.

## 🎯 Features

- **Complete Transformer Architecture**: Implements all key components from the "Attention is All You Need" paper
- **Multi-Head Self-Attention**: Scaled dot-product attention with causal masking for autoregressive generation
- **Pre-Layer Normalization**: Modern architecture following GPT-2/3 design patterns
- **Weight Tying**: Shares weights between token embeddings and output layer for better efficiency
- **Flexible Generation**: Supports temperature sampling and top-k filtering
- **Character-Level Modeling**: Simple character-based tokenization for easy experimentation
- **Training & Inference Scripts**: Ready-to-use scripts for training and text generation

## 🏗️ Architecture

The model implements a decoder-only transformer architecture with the following components:

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
Dropout
    ↓
┌─────────────────────────┐
│  Transformer Block × N  │
│  ┌──────────────────┐   │
│  │ Layer Norm       │   │
│  │ Multi-Head Attn  │   │
│  │ Residual Add     │   │
│  ├──────────────────┤   │
│  │ Layer Norm       │   │
│  │ Feed Forward     │   │
│  │ Residual Add     │   │
│  └──────────────────┘   │
└─────────────────────────┘
    ↓
Layer Norm
    ↓
Linear (to vocab)
    ↓
Output Logits
```

### Key Components

- **MultiHeadAttention**: Implements scaled dot-product attention with multiple attention heads
- **FeedForward**: Position-wise feed-forward network with GELU activation
- **TransformerBlock**: Combines attention and feed-forward with residual connections
- **GPT**: Main model class orchestrating all components

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer.git
cd transformer

# Install dependencies
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- PyTorch 2.0+

## 🚀 Quick Start

### Training

Train a character-level language model on your text data:

```python
# Place your text file as input.txt
python train.py
```

The training script will:
1. Load text from `input.txt` (or use sample text if not found)
2. Build a character vocabulary
3. Train a GPT model with default hyperparameters
4. Save the trained model to `model.pt`

### Generating Text

Generate text using a trained model:

```bash
# Generate with default settings
python generate.py --prompt "To be or not to be"

# Customize generation
python generate.py \
    --prompt "Once upon a time" \
    --max_tokens 500 \
    --temperature 0.8 \
    --top_k 40
```

### Using the Model Programmatically

```python
import torch
from transformer import GPT, GPTConfig, CharDataset

# Load your text
with open('input.txt', 'r') as f:
    text = f.read()

# Create dataset
dataset = CharDataset(text, block_size=128)

# Configure model
config = GPTConfig(
    vocab_size=dataset.vocab_size,
    block_size=128,
    n_layers=4,
    n_heads=4,
    d_model=256,
    d_ff=1024,
    dropout=0.1
)

# Initialize model
model = GPT(config)

# Generate text
prompt = "Your prompt here"
context = torch.tensor([dataset.encode(prompt)], dtype=torch.long)
generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
print(dataset.decode(generated[0].tolist()))
```

## ⚙️ Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | auto | Size of the vocabulary (determined by dataset) |
| `block_size` | 128 | Maximum sequence length / context window |
| `n_layers` | 4 | Number of transformer blocks |
| `n_heads` | 4 | Number of attention heads |
| `d_model` | 256 | Model dimension / embedding size |
| `d_ff` | 1024 | Feed-forward network hidden dimension |
| `dropout` | 0.1 | Dropout probability |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Number of sequences per batch |
| `learning_rate` | 3e-4 | Adam learning rate |
| `num_steps` | 5000 | Total training steps |
| `eval_interval` | 500 | Steps between evaluation logs |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Controls randomness (lower = more conservative) |
| `top_k` | 40 | Limits sampling to top-k most likely tokens |
| `max_new_tokens` | 200 | Maximum tokens to generate |

## 📊 Model Scaling

The default configuration has ~1.5M parameters. You can easily scale the model:

```python
# Small model (~500K parameters)
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=64,
    n_layers=2,
    n_heads=2,
    d_model=128,
    d_ff=512
)

# Large model (~10M parameters)
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=256,
    n_layers=8,
    n_heads=8,
    d_model=512,
    d_ff=2048
)
```

## 📚 Understanding the Code

### Multi-Head Attention

The attention mechanism allows the model to focus on different parts of the input:

```python
# Compute attention: Q @ K^T / sqrt(d_k)
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

# Apply causal mask (prevent looking ahead)
scores = scores.masked_fill(mask, float("-inf"))

# Softmax to get attention weights
attn = F.softmax(scores, dim=-1)

# Apply attention to values
out = attn @ v
```

### Autoregressive Generation

Text is generated one token at a time:

```python
for _ in range(max_new_tokens):
    # Get predictions for next token
    logits = model(context)[:, -1, :]

    # Sample from distribution
    probs = F.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    # Append to context
    context = torch.cat([context, next_token], dim=1)
```

## 🎓 Educational Goals

This implementation is designed to be:

1. **Readable**: Clear variable names and extensive comments
2. **Educational**: Demonstrates core transformer concepts without complex optimizations
3. **Modular**: Each component is self-contained and easy to understand
4. **Hackable**: Simple to modify and experiment with different architectures

## 🔍 Project Structure

```
transformer/
├── transformer.py       # Core model implementation
├── train.py            # Training script
├── generate.py         # Text generation script
├── transformer.ipynb   # Interactive notebook
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## 🎯 Use Cases

- **Learning**: Understand how transformers work from first principles
- **Experimentation**: Test architectural modifications and training techniques
- **Character-level modeling**: Shakespeare generation, code completion, etc.
- **Research**: Baseline implementation for comparing new ideas
- **Education**: Teaching material for ML courses

## 📖 References

This implementation is inspired by:

- ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) - Original transformer paper
- ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT) - Minimal GPT implementation
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for BPE/WordPiece tokenization
- [ ] Implement gradient accumulation for larger effective batch sizes
- [ ] Add learning rate scheduling
- [ ] Support distributed training
- [ ] Add evaluation metrics (perplexity, etc.)
- [ ] Implement KV caching for faster generation
- [ ] Add Flash Attention support

## 📄 License

MIT License - feel free to use this code for learning and experimentation!

## 🙏 Acknowledgments

Special thanks to the PyTorch team and the broader ML research community for making transformers accessible and understandable.

---

**Built with ❤️ for learning and understanding transformers**
