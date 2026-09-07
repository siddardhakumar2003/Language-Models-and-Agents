#!/usr/bin/env python3
"""
Compute ROUGE-L and chrF metrics for Phase 2 evaluation
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except:
    HAS_ROUGE = False
    print("❌ rouge_score not installed. Install: pip install rouge-score")

try:
    import sacrebleu
    HAS_CHRF = True
except:
    HAS_CHRF = False
    print("❌ sacrebleu not installed. Install: pip install sacrebleu")


class RougeChRFComputer:
    """Compute ROUGE-L and chrF metrics."""

    def __init__(self, language: str, model_dir: Path, data_dir: Path):
        self.language = language
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*70}")
        print(f"COMPUTING ROUGE-L & chrF: {language.upper()}")
        print(f"{'='*70}\n")

        self.model = self._load_model()
        self.test_texts = self._load_test_data(max_samples=10000)

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

    def compute_metrics(self) -> Dict:
        """Compute ROUGE-L and chrF."""
        if not self.model or not self.test_texts:
            return {}

        results = {}

        # Generate samples at temperature 1.0 (balanced)
        hypotheses = []
        references = []

        print(f"\nGenerating samples (200 for metrics)...")
        for text in self.test_texts[:200]:
            try:
                # Character-level encoding
                prompt = text[:min(len(text), 50)]
                tokens = [ord(c) for c in prompt]

                if len(tokens) == 0:
                    continue

                input_ids = torch.tensor([tokens], device=self.device)

                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids,
                        max_new_tokens=20,
                        temperature=1.0
                    )

                # Decode back to text
                generated_list = generated[0].cpu().numpy().tolist()
                try:
                    hypothesis = ''.join([chr(c) if c < 1114112 else '?' for c in generated_list])
                    reference = text

                    hypotheses.append(hypothesis)
                    references.append(reference)
                except:
                    pass
            except:
                pass

        print(f"Generated {len(hypotheses)} samples for evaluation")

        # Compute ROUGE-L
        if HAS_ROUGE:
            print(f"\nComputing ROUGE-L...")
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
                rouge_scores = []
                for hyp, ref in zip(hypotheses, references):
                    try:
                        score = scorer.score(ref, hyp)
                        rouge_scores.append(score['rougeL'].fmeasure)
                    except:
                        pass

                if rouge_scores:
                    avg_rouge = np.mean(rouge_scores)
                    results["rouge_l"] = {
                        "fmeasure": round(avg_rouge, 4),
                        "samples": len(rouge_scores)
                    }
                    print(f"✅ ROUGE-L F-measure: {avg_rouge:.4f} (on {len(rouge_scores)} samples)")
            except Exception as e:
                print(f"❌ ROUGE-L error: {e}")
        else:
            print(f"⚠️  ROUGE-L not available")

        # Compute chrF
        if HAS_CHRF:
            print(f"\nComputing chrF...")
            try:
                chrf = sacrebleu.corpus_chrf(hypotheses, [references])
                results["chrf"] = {
                    "score": round(chrf.score, 2),
                    "samples": len(hypotheses)
                }
                print(f"✅ chrF score: {chrf.score:.2f} (on {len(hypotheses)} samples)")
            except Exception as e:
                print(f"❌ chrF error: {e}")
        else:
            print(f"⚠️  chrF not available")

        return results


def main():
    project_root = Path("/media/ubuntu/Personal/IIIT_Hyderabad/Semester_3/LMA/Mini_Project")

    # Load existing metrics
    metrics_file = project_root / "report" / "phase2_complete_metrics.json"
    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    # Compute for each language
    for lang in ["telugu", "bhojpuri"]:
        print(f"\n{'#'*70}")
        print(f"# {lang.upper()}")
        print(f"{'#'*70}")

        computer = RougeChRFComputer(
            lang,
            project_root / lang / "model",
            project_root / lang / "data"
        )

        new_metrics = computer.compute_metrics()

        # Update generation_quality section
        if new_metrics:
            if "generation_quality" not in all_metrics[lang]:
                all_metrics[lang]["generation_quality"] = {}

            all_metrics[lang]["generation_quality"].update(new_metrics)

    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✅ METRICS UPDATED")
    print(f"{'='*70}")
    print(f"File: {metrics_file}\n")

    # Print summary
    for lang, metrics in all_metrics.items():
        print(f"\n{lang.upper()}:")
        if "generation_quality" in metrics:
            gq = metrics["generation_quality"]
            if "bleu" in gq:
                print(f"  BLEU: {gq.get('bleu', 'N/A')}")
            if "rouge_l" in gq:
                print(f"  ROUGE-L: {gq['rouge_l'].get('fmeasure', 'N/A')}")
            if "chrf" in gq:
                print(f"  chrF: {gq['chrf'].get('score', 'N/A')}")


if __name__ == "__main__":
    main()
