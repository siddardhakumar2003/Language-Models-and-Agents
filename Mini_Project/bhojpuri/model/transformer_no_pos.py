"""
Bhojpuri Decoder-Only Transformer -- NO POSITIONAL EMBEDDINGS ablation
(LMA_Individual_Project_v1.pdf, "Bonus (optional): Ablation -- no positional embeddings").

New file. Does NOT modify model/transformer.py. Subclasses the standard BhojpuriTransformer
and overrides only __init__ (skips creating the positional_embedding table) and forward
(skips adding positional embeddings to the token embeddings) -- every other component
(CausalSelfAttention, FeedForward, Block, generate(), count_parameters()) is inherited
unchanged from the standard model, imported, not duplicated.

With no positional information anywhere (no absolute position embedding, no relative
position bias, no RoPE), the only signal the model has about token order is indirect: the
causal mask restricts which positions each query can attend to, so "how many other tokens
are visible" correlates weakly with position, but two different orderings of the same
token *set* into a causally-valid sequence become genuinely difficult for the model to
tell apart from attention alone. This is exactly the effect the bonus ablation is meant to
surface (PDF: "a short explanation of what breaks without position information").
"""

import json
from pathlib import Path
from typing import Optional

import torch.nn as nn

from model.transformer import BhojpuriTransformer, Block


class BhojpuriTransformerNoPos(BhojpuriTransformer):
    """Same architecture as BhojpuriTransformer (same CausalSelfAttention/FeedForward/Block,
    same tied LM head) except there is no positional embedding table at all."""

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        nn.Module.__init__(self)  # bypass BhojpuriTransformer.__init__ (it creates positional_embedding)

        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "model_config_no_pos.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        self.vocab_size = config.get("vocab_size", 16000)
        self.d_model = config.get("embedding_dim", 320)
        self.num_layers = config.get("num_layers", 8)
        self.num_heads = config.get("num_heads", 8)
        self.d_ff = config.get("hidden_dim", 1280)
        self.max_seq_len = config.get("max_seq_length", 256)
        self.dropout = config.get("dropout", 0.1)
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-6)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        # NOTE: no self.positional_embedding -- this omission is the entire ablation.
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList([
            Block(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.ln_final = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, return_attn: bool = False):
        """Identical to BhojpuriTransformer.forward except token embeddings are used directly,
        with no positional embedding added."""
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)

        attn_list = [] if return_attn else None
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits, attn_list if return_attn else None

    # generate() and count_parameters() are inherited unchanged from BhojpuriTransformer --
    # generate() only calls self.forward(), which is correctly overridden above.


if __name__ == "__main__":
    import torch

    print("Testing BhojpuriTransformerNoPos...")
    model = BhojpuriTransformerNoPos()
    print(f"Total parameters: {model.count_parameters():,}")
    assert not hasattr(model, "positional_embedding"), "ablation model must not have a positional_embedding"

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    logits, _ = model(input_ids, return_attn=False)
    assert logits.shape == (batch_size, seq_len, model.vocab_size), "Shape mismatch!"
    print("✓ Forward pass shape OK")

    # The whole point of the ablation: shuffling token order should NOT change per-position
    # logits nearly as much as it would for the standard (positional) model, since there is
    # no position signal to disturb -- only the causal-visibility pattern changes.
    reversed_ids = input_ids.flip(dims=[1])
    logits_rev, _ = model(reversed_ids, return_attn=False)
    print("✓ Ran forward on a reversed-order sequence (sanity check only, no assertion -- "
          "logits legitimately differ since causal masking still makes position 0 vs 9 "
          "structurally different in terms of how many keys are visible).")
    print("✓ All sanity checks passed")
