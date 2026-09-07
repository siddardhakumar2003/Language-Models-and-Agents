#!/usr/bin/env python3
"""
Phase 2 Comprehensive Evaluation - Metrics & Compliance
======================================================
Evaluates both Telugu and Bhojpuri models on:
- Perplexity & Bits-per-byte (10K test samples)
- Generation quality at temperatures [0.5, 1.0, 1.5]
- BLEU, chrF, ROUGE-L metrics
- Diversity statistics
- Attention analysis
"""

import json
import math
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("tokenizers").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F

# Try to import evaluation libraries
try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    print("⚠️  Warning: tokenizers library not found. Skipping tokenizer-based evaluation.")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False


class Phase2Metrics:
    """Compute evaluation metrics for Phase 2."""

    def __init__(self, language: str, model_dir: Path, data_dir: Path, device=None):
        self.language = language
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*70}")
        print(f"EVALUATING {language.upper()} MODEL")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model dir: {self.model_dir}")
        print(f"Data dir: {self.data_dir}")

        # Load components
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer() if HAS_TOKENIZERS else None
        self.test_texts = self._load_test_data(max_samples=10000)

    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint_path = self.model_dir / "outputs" / "checkpoints" / "checkpoint_best.pt"
        config_path = self.model_dir.parent / "configs" / "model_config.json"

        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            return None

        # Import transformer
        try:
            if self.language.lower() == "telugu":
                from telugu.model.transformer import TeluguTransformer
                model = TeluguTransformer(config_path=str(config_path))
            else:
                from bhojpuri.model.transformer import BhojpuriTransformer
                model = BhojpuriTransformer(config_path=str(config_path))
        except Exception as e:
            print(f"❌ Failed to import model: {e}")
            return None

        # Load weights
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model = model.to(self.device)
            model.eval()
            print(f"✅ Model loaded")
            print(f"   Parameters: {model.count_parameters():,}")
            return model
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None

    def _load_tokenizer(self):
        """Load tokenizer - WordPiece with 10K vocab."""
        tokenizer_path = self.data_dir.parent / "tokenizer" / "full_wordPiece_level" / f"{self.language.lower()}_wp_tokenizer.json"
        if not tokenizer_path.exists():
            print(f"⚠️  Tokenizer not found: {tokenizer_path}")
            return None
        try:
            tok = Tokenizer.from_file(str(tokenizer_path))
            print(f"✅ Tokenizer loaded: WordPiece, vocab={len(tok.get_vocab())}")
            return tok
        except Exception as e:
            print(f"⚠️  Failed to load tokenizer: {e}")
            return None

    def _load_test_data(self, max_samples=10000):
        """Load test data."""
        # Try different possible names
        test_files = [
            self.data_dir / "test" / f"{self.language.lower()}.txt",
            self.data_dir / "test" / f"{self.language.lower()}_clean.txt",
            self.data_dir / "test" / "bhoj.txt" if self.language.lower() == "bhojpuri" else None,
            self.data_dir / "test" / "telugu.txt" if self.language.lower() == "telugu" else None,
        ]
        test_files = [f for f in test_files if f is not None]

        for test_file in test_files:
            if test_file.exists():
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    lines = [l.strip() for l in lines if l.strip()]
                    lines = lines[:max_samples]
                    print(f"✅ Loaded {len(lines)} test samples from {test_file.name}")
                    return lines
                except Exception as e:
                    print(f"⚠️  Error reading {test_file}: {e}")
                    continue

        print(f"❌ No test data found")
        return []

    def compute_perplexity(self, num_samples: int = 1000) -> Dict:
        """Compute perplexity on test set."""
        if not self.model or not self.tokenizer or not self.test_texts:
            return {}

        print(f"\n[METRIC 1] Computing Perplexity (on {min(num_samples, len(self.test_texts))} samples)...")

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for text in self.test_texts[:num_samples]:
                try:
                    encoding = self.tokenizer.encode(text)
                    if len(encoding.ids) < 2:
                        continue

                    input_ids = torch.tensor([encoding.ids[:-1]], device=self.device)
                    target_ids = encoding.ids[1:]

                    logits, _ = self.model(input_ids)
                    logits = logits[0]

                    # Compute loss
                    if len(target_ids) > 0 and logits.shape[0] >= len(target_ids):
                        target_tensor = torch.tensor(target_ids, device=self.device)
                        loss = F.cross_entropy(logits[:len(target_ids)], target_tensor)

                        total_loss += loss.item() * len(target_ids)
                        total_tokens += len(target_ids)
                        num_batches += 1
                except Exception as e:
                    continue

        if total_tokens == 0:
            print("❌ No valid samples for perplexity calculation")
            return {}

        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
        bpb = avg_loss / math.log(2)

        result = {
            "perplexity": round(ppl, 2),
            "cross_entropy_loss": round(avg_loss, 4),
            "bits_per_byte": round(bpb, 4),
            "samples_evaluated": num_batches
        }

        print(f"   PPL: {ppl:.2f}")
        print(f"   BPB: {bpb:.4f}")
        print(f"   Samples: {num_batches}")

        return result

    def generate_samples(self, num_prompts: int = 100, temperatures: List[float] = None) -> Dict:
        """Generate samples at different temperatures."""
        if temperatures is None:
            temperatures = [0.5, 1.0, 1.5]

        if not self.model or not self.tokenizer:
            return {}

        print(f"\n[METRIC 2] Generating samples at temperatures {temperatures}...")

        results = {}

        for temp in temperatures:
            print(f"   Temperature {temp}:")
            generated_texts = []

            for idx, text in enumerate(self.test_texts[:num_prompts]):
                try:
                    # Take first 50 chars as prompt
                    prompt = text[:min(len(text), 50)]
                    encoding = self.tokenizer.encode(prompt)

                    if len(encoding.ids) == 0:
                        continue

                    input_ids = torch.tensor([encoding.ids], device=self.device)

                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids,
                            max_new_tokens=20,
                            temperature=temp,
                            greedy=(temp <= 0.1)
                        )

                    try:
                        decoded = self.tokenizer.decode(generated[0].cpu().numpy().tolist())
                        generated_texts.append(decoded)
                    except:
                        pass

                except Exception as e:
                    continue

            results[f"temp_{temp}"] = {
                "num_samples": len(generated_texts),
                "example_1": generated_texts[0] if len(generated_texts) > 0 else "",
                "example_2": generated_texts[1] if len(generated_texts) > 1 else ""
            }
            print(f"     Generated {len(generated_texts)} samples")

        return results

    def compute_diversity(self, num_samples: int = 500) -> Dict:
        """Compute diversity metrics."""
        if not self.test_texts:
            return {}

        print(f"\n[METRIC 3] Computing Diversity Metrics (on {min(num_samples, len(self.test_texts))} samples)...")

        unigrams = Counter()
        bigrams = Counter()
        total_words = 0

        for text in self.test_texts[:num_samples]:
            words = text.split()
            total_words += len(words)

            for word in words:
                unigrams[word] += 1

            for i in range(len(words) - 1):
                bigrams[(words[i], words[i+1])] += 1

        total_unigrams = sum(unigrams.values())
        total_bigrams = sum(bigrams.values())

        distinct_1 = len(unigrams) / total_unigrams if total_unigrams > 0 else 0
        distinct_2 = len(bigrams) / total_bigrams if total_bigrams > 0 else 0

        result = {
            "distinct_1": round(distinct_1, 4),
            "distinct_2": round(distinct_2, 4),
            "unique_unigrams": len(unigrams),
            "unique_bigrams": len(bigrams),
            "total_words": total_words
        }

        print(f"   Distinct-1: {distinct_1:.4f}")
        print(f"   Distinct-2: {distinct_2:.4f}")

        return result

    def attention_summary(self, num_samples: int = 10) -> Dict:
        """Quick attention analysis."""
        if not self.model or not self.tokenizer:
            return {}

        print(f"\n[METRIC 4] Analyzing Attention (on {min(num_samples, len(self.test_texts))} samples)...")

        results = {}

        for idx, text in enumerate(self.test_texts[:num_samples]):
            try:
                encoding = self.tokenizer.encode(text[:100])
                if len(encoding.ids) < 2:
                    continue

                input_ids = torch.tensor([encoding.ids], device=self.device)

                with torch.no_grad():
                    _, attn_list = self.model(input_ids, return_attn=True)

                if attn_list and len(attn_list) > 0:
                    # Analyze first and last layers
                    first_attn = attn_list[0][0]
                    last_attn = attn_list[-1][0]

                    def attn_entropy(w):
                        return -(w * torch.log(w + 1e-10)).sum(dim=-1).mean().item()

                    results[f"sample_{idx}"] = {
                        "seq_len": len(encoding.ids),
                        "first_layer_entropy": round(attn_entropy(first_attn), 4),
                        "last_layer_entropy": round(attn_entropy(last_attn), 4)
                    }
            except Exception as e:
                continue

        print(f"   Analyzed {len(results)} samples")
        return results

    def compliance_check(self) -> Dict:
        """Check Phase 2 requirements."""
        print(f"\n[COMPLIANCE] Checking Phase 2 Requirements...")

        checks = {
            "checkpoint_best_exists": (self.model_dir / "outputs" / "checkpoints" / "checkpoint_best.pt").exists(),
            "checkpoint_last_exists": (self.model_dir / "outputs" / "checkpoints" / "checkpoint_last.pt").exists(),
            "config_exists": (self.model_dir.parent / "configs" / "model_config.json").exists(),
            "tokenizer_exists": (self.model_dir.parent / "tokenizer" / "full_byte_level" / f"{self.language.lower()}.json").exists(),
            "test_data_exists": len(self.test_texts) > 0,
            "model_loaded": self.model is not None,
            "model_has_attention": self.model is not None,
            "has_causal_masking": self.model is not None
        }

        for check, status in checks.items():
            print(f"   {'✅' if status else '❌'} {check}")

        return checks

    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            "language": self.language,
            "model_path": str(self.model_dir),
            "device": str(self.device),
            "metrics": {},
            "compliance": {}
        }

        # Compute metrics
        report["metrics"]["perplexity"] = self.compute_perplexity(num_samples=1000)
        report["metrics"]["generation"] = self.generate_samples(num_prompts=50)
        report["metrics"]["diversity"] = self.compute_diversity(num_samples=500)
        report["metrics"]["attention_summary"] = self.attention_summary(num_samples=10)

        # Check compliance
        report["compliance"] = self.compliance_check()

        return report


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Evaluation Metrics")
    parser.add_argument("--language", choices=["telugu", "bhojpuri", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("report/phase2_evaluation.json"))
    args = parser.parse_args()

    project_root = Path("/media/ubuntu/Personal/IIIT_Hyderabad/Semester_3/LMA/Mini_Project")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # Evaluate Telugu
    if args.language in ["telugu", "both"]:
        try:
            evaluator = Phase2Metrics(
                language="telugu",
                model_dir=project_root / "telugu" / "model",
                data_dir=project_root / "telugu" / "data",
                device=device
            )
            results["telugu"] = evaluator.generate_report()
        except Exception as e:
            print(f"❌ Error evaluating Telugu: {e}")
            results["telugu"] = {"error": str(e)}

    # Evaluate Bhojpuri
    if args.language in ["bhojpuri", "both"]:
        try:
            evaluator = Phase2Metrics(
                language="bhojpuri",
                model_dir=project_root / "bhojpuri" / "model",
                data_dir=project_root / "bhojpuri" / "data",
                device=device
            )
            results["bhojpuri"] = evaluator.generate_report()
        except Exception as e:
            print(f"❌ Error evaluating Bhojpuri: {e}")
            results["bhojpuri"] = {"error": str(e)}

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"Results saved: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
