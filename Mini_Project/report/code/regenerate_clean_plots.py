#!/usr/bin/env python3
"""
Regenerate all Phase 1 plots - CLEAN VERSION (no description boxes)
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')

REPORT_DIR = Path("/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1")
REPORT_DIR.mkdir(exist_ok=True)

# Load data
with open("/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/telugu/data/config.json") as f:
    telugu_config = json.load(f)

with open("/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/bhojpuri/data/config.json") as f:
    bhojpuri_config = json.load(f)

print("="*70)
print("REGENERATING CLEAN PLOTS (no description boxes)")
print("="*70)

# ============================================================================
# Plot 1: Data Collection Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

languages = ['Telugu (Model H)', 'Bhojpuri (Model L)']
total_lines = [
    telugu_config['total_training_lines'],
    bhojpuri_config['total_training_lines']
]
tokens = [
    telugu_config['token_progress']['total_corpus_tokens_estimate'] / 1e6,
    bhojpuri_config['token_progress']['total_corpus_tokens_estimate'] / 1e6
]

colors = ['#1f77b4', '#ff7f0e']
ax1.bar(languages, total_lines, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Total Lines', fontsize=12, fontweight='bold')
ax1.set_title('Data Collection: Total Lines per Language', fontsize=13, fontweight='bold')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
ax1.grid(axis='y', alpha=0.3)

ax2.bar(languages, tokens, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Tokens (Millions)', fontsize=12, fontweight='bold')
ax2.set_title('Estimated Tokens per Language', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_data_collection.png', dpi=300, bbox_inches='tight')
print("✓ plot_data_collection.png")
plt.close()

# ============================================================================
# Plot 2: Cleaning Pass Rates
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

languages = ['Telugu', 'Bhojpuri']
pass_rates = [
    telugu_config['scraping']['pass_rate_percent'],
    bhojpuri_config['scraping']['pass_rate_percent']
]
raw_texts = [
    telugu_config['scraping']['raw_texts'],
    bhojpuri_config['scraping']['raw_texts']
]
cleaned_texts = [
    telugu_config['scraping']['texts_cleaned'],
    bhojpuri_config['scraping']['texts_cleaned']
]

x = np.arange(len(languages))
width = 0.35

bars1 = ax.bar(x - width/2, raw_texts, width, label='Raw Texts', color='#d62728', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, cleaned_texts, width, label='Cleaned Texts', color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Number of Texts', fontsize=12, fontweight='bold')
ax.set_title('Data Cleaning Pipeline Results', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_cleaning_results.png', dpi=300, bbox_inches='tight')
print("✓ plot_cleaning_results.png")
plt.close()

# ============================================================================
# Plot 3: Train/Val/Test Split Distribution
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

te_split_labels = ['Train', 'Validation', 'Test']
te_split_sizes = [
    telugu_config['splits']['train']['lines'],
    telugu_config['splits']['val']['lines'],
    telugu_config['splits']['test']['lines']
]

colors_split = ['#2ecc71', '#3498db', '#e74c3c']
wedges1, texts1, autotexts1 = ax1.pie(te_split_sizes, labels=te_split_labels, autopct='%1.1f%%',
                                       colors=colors_split, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Telugu (Model H) - Data Split\nTotal: 25.3M lines', fontsize=12, fontweight='bold')

bho_split_labels = ['Train', 'Validation', 'Test']
bho_split_sizes = [
    bhojpuri_config['splits']['train']['lines'],
    bhojpuri_config['splits']['val']['lines'],
    bhojpuri_config['splits']['test']['lines']
]

wedges2, texts2, autotexts2 = ax2.pie(bho_split_sizes, labels=bho_split_labels, autopct='%1.1f%%',
                                       colors=colors_split, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
total_bho = sum(bho_split_sizes)
ax2.set_title(f'Bhojpuri (Model L) - Data Split\nTotal: {total_bho/1e6:.2f}M lines', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_data_splits.png', dpi=300, bbox_inches='tight')
print("✓ plot_data_splits.png")
plt.close()

# ============================================================================
# Plot 4: Data Sources (Bhojpuri)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

sources = ['HuggingFace\nCorpus', 'OCR\nAugmentation', 'English→Bhojpuri\nTranslation', 'Hindi→Bhojpuri\nTranslation', 'Web\nScraping']
source_lines = [261000, 511599, 711138, 15852, 3234]

ax.barh(sources, source_lines, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Lines', fontsize=12, fontweight='bold')
ax.set_title('Bhojpuri Data Sources Composition', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
ax.grid(axis='x', alpha=0.3)

for i, (src, val) in enumerate(zip(sources, source_lines)):
    pct = (val / sum(source_lines)) * 100
    ax.text(val, i, f'  {val:,.0f}', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_data_sources_bhojpuri.png', dpi=300, bbox_inches='tight')
print("✓ plot_data_sources_bhojpuri.png")
plt.close()

# ============================================================================
# Plot 5: Token Progress
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

languages = ['Telugu\n(Model H)', 'Bhojpuri\n(Model L)']
current_tokens = [
    telugu_config['token_progress']['total_corpus_tokens_estimate'] / 1e6,
    bhojpuri_config['token_progress']['total_corpus_tokens_estimate'] / 1e6
]
target = 500

bars = ax.barh(languages, current_tokens, height=0.6, color=['#2ecc71', '#e74c3c'],
               alpha=0.8, edgecolor='black', linewidth=2)

ax.axvline(target, color='red', linestyle='--', linewidth=2.5, label='Target (500M)', zorder=10)

for i, (lang, tokens) in enumerate(zip(languages, current_tokens)):
    pct = (tokens / target) * 100
    ax.text(tokens + 20, i, f'{tokens:.1f}M ({pct:.1f}%)', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Tokens (Millions)', fontsize=12, fontweight='bold')
ax.set_title('Progress Toward 500M Token Target', fontsize=13, fontweight='bold')
ax.set_xlim(0, 600)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_token_progress.png', dpi=300, bbox_inches='tight')
print("✓ plot_token_progress.png")
plt.close()

# ============================================================================
# Plot 6: Tokenizer Vocab Size vs Unique Tokens
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(5)
width = 0.35

vocab_sizes = [50000, 50000, 50000, 8000, 32000]
unique_tokens = [1978, 11398, 9539, 7761, 31258]
labels = ['Telugu\nByte-level', 'Telugu\nUnicode-level', 'Telugu\nWordPiece',
          'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode-level']

bars1 = ax.bar(x_pos - width/2, vocab_sizes, width, label='Vocabulary Size',
              color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, unique_tokens, width, label='Unique Tokens Used',
              color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Token Count', fontsize=12, fontweight='bold')
ax.set_title('Tokenizer Efficiency: Vocabulary Size vs Actual Usage', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_tokenizer_vocab.png', dpi=300, bbox_inches='tight')
print("✓ plot_tokenizer_vocab.png")
plt.close()

# ============================================================================
# Plot 7: Tokenizer Fertility
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

fertility_data = [1.3327, 5.9283, 5.8353, 1.5412, 4.348, 3.95]
labels_fert = ['Telugu\nByte-level', 'Telugu\nUnicode-level', 'Telugu\nWordPiece',
               'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode-level', 'Bhojpuri\nWordPiece']

colors_fertility = ['#2ca02c' if x < 3 else '#d62728' for x in fertility_data]
bars = ax.bar(labels_fert, fertility_data, color=colors_fertility, alpha=0.8,
             edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Average Characters per Token', fontsize=12, fontweight='bold')
ax.set_title('Tokenizer Fertility: Compression Efficiency', fontsize=13, fontweight='bold')
ax.set_ylim(0, 7)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_tokenizer_fertility.png', dpi=300, bbox_inches='tight')
print("✓ plot_tokenizer_fertility.png")
plt.close()

# ============================================================================
# Plot 8: Unknown Token Rates
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

unk_rates = [0.0, 0.0, 0.0, 0.0, 0.0]
unk_rates_pct = [r * 100 for r in unk_rates]
labels_unk = ['Telugu\nByte-level', 'Telugu\nUnicode-level', 'Telugu\nWordPiece',
              'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode-level']

colors_unk = ['#2ecc71' if x < 0.0001 else '#ff7f0e' for x in unk_rates_pct]
bars = ax.bar(labels_unk, unk_rates_pct, color=colors_unk, alpha=0.8,
             edgecolor='black', linewidth=1.5)

# Add a small marker to show where 0.0002% would be (Bhojpuri WP)
ax.axhline(y=0.00002, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2.5, 0.00003, 'Bhojpuri WP: 0.0002%', fontsize=9, fontweight='bold', color='orange')

ax.set_ylabel('Unknown Token Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Unknown Token Rates: Coverage Quality', fontsize=13, fontweight='bold')
ax.set_ylim(-0.00005, 0.00005)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_tokenizer_unk_rate.png', dpi=300, bbox_inches='tight')
print("✓ plot_tokenizer_unk_rate.png")
plt.close()

# ============================================================================
# Plot 9: Vocabulary Coverage
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

coverage_data = [99.0, 22.8, 19.1, 97.01, 97.68]
labels_cov = ['Telugu\nByte-level', 'Telugu\nUnicode-level', 'Telugu\nWordPiece',
              'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode-level']

colors_cov = ['#2ecc71' if x > 90 else '#ff7f0e' for x in coverage_data]
bars = ax.bar(labels_cov, coverage_data, color=colors_cov, alpha=0.8,
             edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% Threshold', alpha=0.7)
ax.set_ylabel('Vocabulary Coverage (%)', fontsize=12, fontweight='bold')
ax.set_title('Vocabulary Coverage on Test Sets', fontsize=13, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_tokenizer_coverage.png', dpi=300, bbox_inches='tight')
print("✓ plot_tokenizer_coverage.png")
plt.close()

# ============================================================================
# Plot 10: Tokenizer Fertility (Compression Efficiency)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

fertility_all = [1.3327, 5.9283, 5.8353, 1.5412, 4.348]
labels_all = ['Telugu\nByte-level', 'Telugu\nUnicode-level', 'Telugu\nWordPiece',
              'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode-level']

colors = ['#2ca02c' if x < 3 else '#d62728' for x in fertility_all]
bars = ax.bar(labels_all, fertility_all, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, fertility_all):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add compression categories
ax.axhline(y=3, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Compression threshold')
ax.text(0.5, 3.2, 'High Compression', fontsize=10, fontweight='bold', color='green', alpha=0.7)
ax.text(2.5, 3.2, 'Linguistic Units', fontsize=10, fontweight='bold', color='red', alpha=0.7)

ax.set_ylabel('Average Characters per Token', fontsize=13, fontweight='bold')
ax.set_title('Tokenizer Fertility: Compression Efficiency', fontsize=14, fontweight='bold')
ax.set_ylim(0, 7)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(REPORT_DIR / 'plot_tokenizer_analysis.png', dpi=300, bbox_inches='tight')
print("✓ plot_tokenizer_analysis.png (fertility only)")
plt.close()

print("\n" + "="*70)
print("✓ ALL PLOTS REGENERATED (CLEAN VERSION - NO DESCRIPTION BOXES)")
print("="*70)
print(f"\nLocation: {REPORT_DIR}")
print("\nGenerated 10 plots:")
print("  ✓ plot_data_collection.png")
print("  ✓ plot_cleaning_results.png")
print("  ✓ plot_data_splits.png")
print("  ✓ plot_data_sources_bhojpuri.png")
print("  ✓ plot_token_progress.png")
print("  ✓ plot_tokenizer_vocab.png")
print("  ✓ plot_tokenizer_fertility.png")
print("  ✓ plot_tokenizer_unk_rate.png")
print("  ✓ plot_tokenizer_coverage.png")
print("  ✓ plot_tokenizer_analysis.png")
print("="*70)
