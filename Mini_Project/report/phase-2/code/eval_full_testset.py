#!/usr/bin/env python3
"""
Compute metrics on full test set (5000+ samples) for accurate PPL
"""

import json
import math
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except:
    HAS_ROUGE = False

try:
    import sacrebleu
    HAS_CHRF = True
except:
    HAS_CHRF = False


class FullTestsetEvaluator:
    """Evaluate on full test set."""

    def __init__(self, language: str, model_dir: Path, data_dir: Path):
        self.language = language
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*70}")
        print(f"FULL TESTSET EVALUATION: {language.upper()}")
        print(f"{'='*70}\n")

        self.model = self._load_model()
        self.test_texts = self._load_test_data(max_samples=10000)
        self.results = {}

    def _load_model(self):
        """Load model."""
        checkpoint_path = self.model_dir / "outputs" / "checkpoints" / "checkpoint_best.pt"
        config_path = self.model_dir.parent / "configs" / "model_config.json"

        if not checkpoint_path.exists() or not config_path.exists():
            return None

        try:
            if self.language.lower() == "telugu":
                from telugu.model.transformer import TeluguTransformer
                model = TeluguTransformer(config_path=str(config_path))
            else:
                from bhojpuri.model.transformer import BhojpuriTransformer
                model = BhojpuriTransformer(config_path=str(config_path))

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()
            print(f"✅ Model loaded")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

    def _load_test_data(self, max_samples=10000):
        """Load test data."""
        test_files = [
            self.data_dir / "test" / f"{self.language.lower()}.txt",
            self.data_dir / "test" / "bhoj.txt" if self.language.lower() == "bhojpuri" else None,
            self.data_dir / "test" / "telugu.txt" if self.language.lower() == "telugu" else None,
        ]
        test_files = [f for f in test_files if f is not None]

        for test_file in test_files:
            if test_file.exists():
                with open(test_file, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                lines = lines[:max_samples]
                print(f"✅ Loaded {len(lines)} test samples")
                return lines

        return []

    def compute_perplexity_full(self):
        """Compute PPL on full test set."""
        if not self.model or not self.test_texts:
            return {}

        total_loss = 0.0
        total_tokens = 0
        num_samples = 0
        batch_size = 100

        print(f"\nComputing perplexity on {len(self.test_texts)} test samples...")

        # Process in batches
        for batch_idx in range(0, len(self.test_texts), batch_size):
            batch = self.test_texts[batch_idx:batch_idx+batch_size]

            for text in batch:
                try:
                    tokens = [ord(c) for c in text[:128]]
                    if len(tokens) < 2:
                        continue

                    input_ids = torch.tensor([tokens[:-1]], device=self.device)
                    target_ids = torch.tensor(tokens[1:], device=self.device)

                    with torch.no_grad():
                        logits, _ = self.model(input_ids)

                    if logits.shape[0] > 0 and logits.shape[1] >= len(target_ids):
                        loss = F.cross_entropy(logits[0, :len(target_ids)], target_ids)
                        total_loss += loss.item() * len(target_ids)
                        total_tokens += len(target_ids)
                        num_samples += 1
                except:
                    pass

            if batch_idx % 500 == 0:
                print(f"  Processed {batch_idx}/{len(self.test_texts)} samples...")

        if total_tokens == 0:
            print(f"   ❌ Failed to compute perplexity")
            return {}

        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20))
        bpb = avg_loss / math.log(2)

        result = {
            "perplexity": round(ppl, 2),
            "cross_entropy_loss": round(avg_loss, 4),
            "bits_per_byte": round(bpb, 4),
            "samples_evaluated": num_samples,
            "total_tokens": total_tokens
        }
        print(f"   ✅ PPL: {ppl:.2f} | BPB: {bpb:.4f} | Samples: {num_samples} | Tokens: {total_tokens}")
        return result


def main():
    project_root = Path("/media/ubuntu/Personal/IIIT_Hyderabad/Semester_3/LMA/Mini_Project")

    # Load existing metrics
    metrics_file = project_root / "report" / "phase2_complete_metrics.json"
    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    # Compute for each language on full test set
    for lang in ["telugu", "bhojpuri"]:
        print(f"\n{'#'*70}")
        print(f"# {lang.upper()} - FULL TESTSET")
        print(f"{'#'*70}")

        evaluator = FullTestsetEvaluator(
            lang,
            project_root / lang / "model",
            project_root / lang / "data"
        )

        new_ppl = evaluator.compute_perplexity_full()

        # Update perplexity section
        if new_ppl:
            all_metrics[lang]["perplexity"] = new_ppl

    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✅ METRICS UPDATED WITH FULL TESTSET EVALUATION")
    print(f"{'='*70}\n")

    # Print summary
    for lang, metrics in all_metrics.items():
        print(f"\n{lang.upper()} (Full Testset):")
        if "perplexity" in metrics:
            ppl = metrics["perplexity"]
            print(f"  Perplexity: {ppl.get('perplexity', 'N/A')}")
            print(f"  Samples: {ppl.get('samples_evaluated', 'N/A')}")
            print(f"  Tokens: {ppl.get('total_tokens', 'N/A')}")


if __name__ == "__main__":
    main()
