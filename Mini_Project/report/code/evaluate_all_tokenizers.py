#!/usr/bin/env python3
"""
Comprehensive Tokenizer Evaluation for Phase 1 Report
Evaluates all 6 tokenizers (3 Telugu, 3 Bhojpuri) on held-out test sets
"""

import json
import os
from pathlib import Path
from tokenizers import Tokenizer
from collections import Counter
import statistics

# Configuration
PROJECT_ROOT = Path("/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project")
REPORT_DIR = PROJECT_ROOT / "report" / "phase-1"

TOKENIZERS = {
    "Telugu": {
        "byte_level": PROJECT_ROOT / "telugu/tokenizer/full_byte_level/telugu_tokenizer_full.json",
        "unicode_level": PROJECT_ROOT / "telugu/tokenizer/full_unicode_level/checkpoint_batch_140.json",
        "wordpiece": PROJECT_ROOT / "telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json",
    },
    "Bhojpuri": {
        "byte_level": PROJECT_ROOT / "bhojpuri/tokenizer/full_byte_level/bhoj_tokenizer_report_full.json",  # Will use actual tokenizer file
        "unicode_level": PROJECT_ROOT / "bhojpuri/tokenizer/full_unicode_level/bhoj_tokenizer_report_full.json",
        "wordpiece": PROJECT_ROOT / "bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json",
    }
}

# Load actual tokenizer files (not reports)
TOKENIZER_FILES = {
    "Telugu": {
        "byte_level": PROJECT_ROOT / "telugu/tokenizer/full_byte_level/telugu_tokenizer_full.json",
        "unicode_level": PROJECT_ROOT / "telugu/tokenizer/full_unicode_level/checkpoint_batch_140.json",
        "wordpiece": PROJECT_ROOT / "telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json",
    },
    "Bhojpuri": {
        "byte_level": PROJECT_ROOT / "bhojpuri/tokenizer/full_byte_level/bhoj_tokenizer_report_full.json",
        "unicode_level": PROJECT_ROOT / "bhojpuri/tokenizer/full_unicode_level/bhoj_tokenizer_report_full.json",
        "wordpiece": PROJECT_ROOT / "bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json",
    }
}

def load_test_data(language):
    """Load test set for a language"""
    if language == "Telugu":
        test_dir = PROJECT_ROOT / "telugu/data/test"
        files = list(test_dir.glob("*.txt"))
    else:
        test_dir = PROJECT_ROOT / "bhojpuri/data/test"
        files = list(test_dir.glob("*.txt"))

    texts = []
    for f in files[:1]:  # Use first file for efficiency
        try:
            with open(f, 'r', encoding='utf-8') as file:
                texts.extend([line.strip() for line in file if line.strip()][:500])  # Sample for speed
        except:
            pass

    return texts[:500]  # Sample 500 lines for evaluation

def evaluate_tokenizer(tokenizer_path, test_texts, tokenizer_name, language):
    """Evaluate a single tokenizer"""
    print(f"\n{'='*70}")
    print(f"Evaluating {language} - {tokenizer_name}")
    print(f"{'='*70}")

    try:
        # Load tokenizer
        if not tokenizer_path.exists():
            print(f"❌ Tokenizer file not found: {tokenizer_path}")
            return None

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Evaluation metrics
        all_tokens = []
        all_token_ids = []
        unknown_count = 0
        text_lengths = []
        token_lengths = []

        # Process test texts
        for text in test_texts:
            if not text.strip():
                continue

            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            token_ids = encoding.ids

            # Count unknowns
            unk_token = "[UNK]"
            unknown_count += tokens.count(unk_token)

            all_tokens.extend(tokens)
            all_token_ids.extend(token_ids)
            text_lengths.append(len(text))
            token_lengths.append(len(tokens))

        if not all_tokens:
            print("❌ No tokens generated")
            return None

        # Calculate statistics
        total_chars = sum(text_lengths)
        total_tokens = len(all_tokens)
        avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
        unique_tokens = len(set(all_tokens))
        unk_rate = (unknown_count / total_tokens * 100) if total_tokens > 0 else 0

        # Token frequency
        token_freq = Counter(all_tokens)
        top_tokens = token_freq.most_common(10)

        # Vocabulary size
        vocab_size = tokenizer.get_vocab_size()

        # Tokenization examples
        example_texts = test_texts[:3]
        examples = []
        for text in example_texts:
            encoding = tokenizer.encode(text)
            examples.append({
                "text": text[:100],
                "tokens": encoding.tokens[:20],
                "num_tokens": len(encoding.tokens)
            })

        stats = {
            "tokenizer": tokenizer_name,
            "language": language,
            "vocabulary_size": vocab_size,
            "unique_tokens_used": unique_tokens,
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "avg_chars_per_token": round(avg_chars_per_token, 4),
            "unknown_token_count": unknown_count,
            "unknown_token_rate": round(unk_rate, 4),
            "token_frequency_top_10": [(token, count) for token, count in top_tokens],
            "tokenization_examples": examples
        }

        # Print summary
        print(f"✓ Vocabulary Size: {vocab_size:,}")
        print(f"✓ Unique Tokens Used: {unique_tokens:,}")
        print(f"✓ Total Tokens: {total_tokens:,}")
        print(f"✓ Average Chars/Token: {avg_chars_per_token:.4f}")
        print(f"✓ Unknown Token Rate: {unk_rate:.4f}%")
        print(f"✓ Unknown Token Count: {unknown_count}")
        print(f"\nTop 10 Token Frequencies:")
        for token, count in top_tokens:
            print(f"  {token:30} {count:10,} ({count/total_tokens*100:5.2f}%)")

        return stats

    except Exception as e:
        print(f"❌ Error evaluating tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TOKENIZER EVALUATION")
    print("Phase 1 Report - All 6 Tokenizers")
    print("="*70)

    results = {}

    for language in ["Telugu", "Bhojpuri"]:
        print(f"\n{'#'*70}")
        print(f"# {language}")
        print(f"{'#'*70}")

        # Load test data
        test_texts = load_test_data(language)
        print(f"Loaded {len(test_texts)} test samples")

        results[language] = {}

        # Evaluate each tokenizer variant
        for variant_name, tokenizer_path in TOKENIZER_FILES[language].items():
            stats = evaluate_tokenizer(tokenizer_path, test_texts, variant_name, language)
            if stats:
                results[language][variant_name] = stats

    # Save comprehensive report
    report_path = REPORT_DIR / "tokenizer_evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✓ Report saved to: {report_path}")
    print(f"{'='*70}\n")

    # Generate LaTeX table data
    generate_latex_table(results)

def generate_latex_table(results):
    """Generate LaTeX table for report"""
    latex_file = REPORT_DIR / "tokenizer_stats_latex.txt"

    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("% Tokenizer Evaluation Results\n\n")

        for language in ["Telugu", "Bhojpuri"]:
            f.write(f"% {language}\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write("\\begin{tabular}{|l|r|r|r|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Variant} & \\textbf{Vocab} & \\textbf{Avg Chars/Token} & \\textbf{UNK Rate} & \\textbf{Unique Tokens} \\\\\n")
            f.write("\\hline\n")

            for variant, stats in results[language].items():
                if stats:
                    vocab = f"{stats['vocabulary_size']:,}"
                    avg_chars = f"{stats['avg_chars_per_token']:.4f}"
                    unk_rate = f"{stats['unknown_token_rate']:.4f}%"
                    unique = f"{stats['unique_tokens_used']:,}"

                    f.write(f"{variant} & {vocab} & {avg_chars} & {unk_rate} & {unique} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{{language} Tokenizer Evaluation Results}}\n")
            f.write(f"\\label{{tab:tokenizer_eval_{language.lower()}}}\n")
            f.write("\\end{table}\n\n")

    print(f"✓ LaTeX table generated: {latex_file}")

if __name__ == "__main__":
    main()
