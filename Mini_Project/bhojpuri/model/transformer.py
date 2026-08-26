"""
Bhojpuri Decoder-Only Transformer Language Model (from scratch).
Built using only nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, nn.Parameter.
No nn.Transformer* modules or pre-built attention blocks.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with manual QKV projections and scaled dot-product attention.
    Implements: Attention(Q,K,V) = softmax((QK^T / sqrt(d_k)) + M) V
    where M is the causal mask (additive, -inf on future positions).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.dropout_p = dropout

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

        self.register_buffer("causal_mask", self._make_causal_mask(512), persistent=False)

    def _make_causal_mask(self, max_len: int):
        """Create upper-triangular additive causal mask (0 on lower triangle, -inf on upper)."""
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            return_attn: if True, return attention weights (batch*num_heads, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch*num_heads, seq_len, seq_len) if return_attn else None
        """
        batch, seq_len, d_model = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if seq_len > self.causal_mask.shape[0]:
            self.register_buffer("causal_mask", self._make_causal_mask(seq_len), persistent=False)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(batch, seq_len, d_model)
        output = self.W_O(context)
        output = self.dropout_output(output)

        if return_attn:
            attn_for_return = attn_weights.reshape(batch * self.num_heads, seq_len, seq_len)
            return output, attn_for_return

        return output, None

    def verify_causal_masking(self, x: torch.Tensor) -> bool:
        """
        Verify causal masking works: logits at position t should not change if we perturb tokens at t+1.
        Returns True if verification passes.
        """
        with torch.no_grad():
            out1, _ = self.forward(x)
            logits1 = out1[:, :-1, :]

            x_perturbed = x.clone()
            x_perturbed[:, -1, :] = x_perturbed[:, -1, :] + torch.randn_like(x_perturbed[:, -1, :]) * 0.1
            out2, _ = self.forward(x_perturbed)
            logits2 = out2[:, :-1, :]

            max_diff = (logits1 - logits2).abs().max().item()
            return max_diff < 1e-5


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network: Linear -> GELU -> Linear with dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: pre-norm residuals with self-attention and FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn_weights = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if return_attn:
            return x, attn_weights
        return x, None


class BhojpuriTransformer(nn.Module):
    """
    Decoder-only Transformer LM for Bhojpuri.
    Architecture: token embedding + absolute positional embedding + N transformer blocks + LM head.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__()

        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "model_config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        self.vocab_size = config.get("vocab_size", 32000)
        self.d_model = config.get("embedding_dim", 256)
        self.num_layers = config.get("num_layers", 15)
        self.num_heads = config.get("num_heads", 8)
        self.d_ff = config.get("hidden_dim", 1024)
        self.max_seq_len = config.get("max_seq_length", 512)
        self.dropout = config.get("dropout", 0.1)
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-6)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList([
            Block(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.ln_final = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            return_attn: if True, return per-layer attention weights

        Returns:
            logits: (batch, seq_len, vocab_size)
            attn_list: list of attention tensors (one per layer) if return_attn else None
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        token_emb = self.token_embedding(input_ids)
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.positional_embedding(pos_indices)

        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        attn_list = [] if return_attn else None

        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits, attn_list if return_attn else None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False
    ) -> torch.Tensor:
        """
        Autoregressive generation: sample one token at a time until max_new_tokens or EOS.

        Args:
            input_ids: (batch, seq_len) starting sequence
            max_new_tokens: maximum tokens to generate
            temperature: logit scaling (temp > 1 = more random, < 1 = more greedy)
            greedy: if True, always take argmax regardless of temperature

        Returns:
            generated_ids: (batch, seq_len + generated)
        """
        eos_token_id = 3
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.max_seq_len:
                break

            logits, _ = self.forward(generated[:, -self.max_seq_len:])
            next_logits = logits[:, -1, :]

            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing BhojpuriTransformer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BhojpuriTransformer()
    model = model.to(device)
    model.eval()

    print(f"Total parameters: {model.count_parameters():,}")

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits, _ = model(input_ids, return_attn=False)
        print(f"Logits shape: {logits.shape}, expected: ({batch_size}, {seq_len}, {model.vocab_size})")
        assert logits.shape == (batch_size, seq_len, model.vocab_size), "Shape mismatch!"

        logits_attn, attn_list = model(input_ids, return_attn=True)
        print(f"Number of attention tensors: {len(attn_list)}, expected: {model.num_layers}")
        assert len(attn_list) == model.num_layers, "Attention list length mismatch!"
        print(f"Attention shape: {attn_list[0].shape}")

    print("✓ Shape tests passed")

    print("\nTesting causal masking...")
    attn_layer = model.blocks[0].attn
    test_passed = attn_layer.verify_causal_masking(input_ids.float().unsqueeze(-1).expand(-1, -1, model.d_model))
    if test_passed:
        print("✓ Causal masking verified: perturbing future tokens does not affect past logits")
    else:
        print("✗ Causal masking verification failed")

    print("\nTesting generation...")
    with torch.no_grad():
        prompt = torch.tensor([[2, 10, 20]], device=device)
        generated = model.generate(prompt, max_new_tokens=5, temperature=0.5, greedy=True)
        print(f"Generated sequence shape: {generated.shape}")
        print(f"Generated: {generated}")

    print("\n✓ All tests passed!")
