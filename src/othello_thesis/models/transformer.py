# Transformer architecture adapted for Othello move prediction.
from logging import config
from statistics import mode

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class GPTConfig:
    """Config for a minGPT-style Transformer on Othello.
    
    Set board_size explicitly for your experiment (8, 12, or 16).
    vocab_size and block_size are derived from board_size in __post_init__
    unless set explicitly.
    """
    # Board-size parameter (drives vocab_size and block_size)
    board_size: int = 8

    # Architecture
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    
    # Regularization
    dropout: float = 0.1
    
    # Derived fields (computed in __post_init__ if left as None)
    vocab_size: Optional[int] = None
    block_size: Optional[int] = None
    
    def __post_init__(self):
        # For an n×n Othello board:
        #   - max_moves = n² - 4 (subtract 4 starting positions)
        #   - vocab = max_moves + 1 (the +1 is the input padding token,
        #     placed at index = max_moves, used to fill short games)
        #   - block_size = max_moves - 1 (next-token prediction shift)
        max_moves = self.board_size ** 2 - 4
        if self.vocab_size is None:
            self.vocab_size = max_moves + 1  # +1 for pass token
        if self.block_size is None:
            self.block_size = max_moves - 1
        
        # Sanity checks
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with combined QKV projection.
    
    Follows the minGPT/Karpathy convention. Uses a pre-computed causal mask
    stored as a buffer, and two dropouts (attention and residual).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        # Store hyperparameters from config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads

        # Combined QKV projection for efficiency (one matmul instead of three)
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask as a non-trainable buffer
        # Shape (1, 1, block_size, block_size) for broadcasting with
        # attention scores of shape (B, n_heads, T, T)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        """
        Forward pass of causal self-attention.

        Args:
            x: input tensor of shape (B, T, C)
            B = batch size, T = sequence length, C = d_model

        Returns:
            out: output tensor of shape (B, T, C)
        """

        # We need B, T, C for later reshape operations.
        B, T, C = x.size()

        # Apply the c_attn linear layer to project x into queries, keys, values.
        qkv = self.c_attn(x)
        
        # Split into separate Q, K, V tensors
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Each of q, k, v needs to go from (B, T, C) to (B, n_heads, T, d_head).
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)


        # Compute attention scores with scaling
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax over the last dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Attention dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum over values
        out = attn_weights @ v
        
        # Combine heads back into a single tensor
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.c_proj(out)

        # Residual dropout
        out = self.resid_dropout(out)



        return out



class MLP(nn.Module):
    """Position-wise feedforward network with GELU activation.
    
    Standard Transformer MLP: expand 4×, apply GELU, project back.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)   

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        # Apply Feedforward network
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x) 

        return x


class TransformerBlock(nn.Module):
    """A single Transformer block (pre-norm).
    
    Applies layer norm → attention → residual,
    then layer norm → MLP → residual.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()

        # Define four submodules:
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        
        # Pre-norm pattern:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))   

        return x


class OthelloGPT(nn.Module):
    """GPT-style Transformer for Othello move prediction.
    
    Consists of an embedding layer, a stack of Transformer blocks, and a
    final linear layer to produce logits over the move vocabulary.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding for move indices
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        # Positional embedding for sequence positions
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        # Dropout after embeddings
        self.drop = nn.Dropout(config.dropout)
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        # Final layer norm before output
        self.ln_f = nn.LayerNorm(config.d_model)
        # Output projection to vocab size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Standard init for all modules
        self.apply(self._init_weights)
        # Scaled init for c_proj layers (residual projections)
        # This keeps the residual stream from blowing up in deep models.
        # Formula: std / sqrt(2 * n_layers), where 2 accounts for attention + MLP
        # contributions per layer.
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                std = 0.02 / math.sqrt(2 * self.config.n_layers)
                torch.nn.init.normal_(param, mean=0.0, std=std)

    def _init_weights(self, module):
        """Custom weight initialization for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) tensor of token indices (move indices)
            targets: (B, T) tensor of target token indices, or None
        Returns:
            logits: (B, T, vocab_size) tensor of logits
            loss: scalar cross-entropy loss, or None if targets is None
        """
        B, T = idx.size()

        # Position vector
        pos = torch.arange(0, T, device=idx.device)

        # Embed tokens and positions, then apply dropout
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # If targets are provided, compute cross-entropy loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None

        return logits, loss

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 50)
    print("Smoke test for OthelloGPT")
    print("=" * 50)
    
    # Test 1: Construction
    config = GPTConfig(board_size=8, n_layers=2, n_heads=4, d_model=64, dropout=0.1)
    model = OthelloGPT(config)
    print(f"✓ Model constructed")
    
    # Test 2: Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Test 3: Forward without targets
    B, T = 4, config.block_size
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, config.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss is None, "Loss should be None when targets not provided"
    print(f"✓ Forward without targets: logits {logits.shape}, loss=None")
    
    # Test 4: Forward with targets
    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, config.vocab_size)
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    expected_loss = math.log(config.vocab_size)
    print(f"✓ Forward with targets: loss={loss.item():.4f} (expected ≈ {expected_loss:.4f})")
    
    # Test 5: Backward
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total, f"Only {n_with_grad}/{n_total} parameters have gradients"
    
    # Check gradients are not NaN
    has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    assert not has_nan_grad, "Some gradients are NaN!"
    print(f"✓ Backward: {n_with_grad}/{n_total} parameters have valid gradients")
    
    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)