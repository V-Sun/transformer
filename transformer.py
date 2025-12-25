"""
GPT-style Transformer implementation from scratch using PyTorch.

This module implements a decoder-only transformer architecture similar to GPT.
Features:
- Multi-head self-attention with causal masking
- Position and token embeddings
- Layer normalization (Pre-LN architecture)
- Feed-forward networks with GELU activation
- Text generation with temperature and top-k sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT model"""
    vocab_size: int
    block_size: int
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with causal masking"""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Query, Key, Value projections
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        """
        Args:
            x: Input tensor of shape (B, T, D)
            causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.wq(x)  # (B, T, D)
        k = self.wk(x)  # (B, T, D)
        v = self.wv(x)  # (B, T, D)

        # Split into multiple heads: (B, T, D) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores: (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask (prevent attending to future tokens)
        if causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask.view(1, 1, T, T), float("-inf"))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        attn = self.attn_drop(attn)

        # Apply attention to values: (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        out = attn @ v

        # Concatenate heads: (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        out = self.resid_drop(self.wo(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-layer normalization"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        # Pre-LN architecture with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style decoder-only transformer model"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Output layers
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (share weights between token embedding and output layer)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx: Token indices of shape (B, T)
            targets: Target token indices of shape (B, T), optional

        Returns:
            logits: Logits of shape (B, T, vocab_size)
            loss: Cross-entropy loss (if targets provided), else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, D)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, D)

        # Combine embeddings and apply dropout
        x = self.dropout(tok_emb + pos_emb)  # (B, T, D)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)  # (B, T, D)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Conditioning sequence of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


class CharDataset:
    """Simple character-level dataset for text generation"""

    def __init__(self, text, block_size):
        self.block_size = block_size

        # Build character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode entire text
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def encode(self, text):
        """Encode text to token indices"""
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        """Decode token indices to text"""
        return ''.join([self.itos[i] for i in indices])

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size):
        """
        Get a random batch of training data.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            x: Input sequences of shape (batch_size, block_size)
            y: Target sequences of shape (batch_size, block_size)
        """
        ix = torch.randint(len(self) - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y
