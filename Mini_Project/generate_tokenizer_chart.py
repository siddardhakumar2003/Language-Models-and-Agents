import matplotlib.pyplot as plt
import numpy as np

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Vocabulary Size Comparison
languages = ['Telugu\nByte-level', 'Telugu\nUnicode', 'Telugu\nWordPiece',
             'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode', 'Bhojpuri\nWordPiece']
vocab_sizes = [50000, 50000, 50000, 8000, 16000, 16000]
unique_tokens = [1978, 11398, 9539, 7761, 15643, 15000]  # approximate for WP

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']

x_pos = np.arange(len(languages))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, vocab_sizes, width, label='Vocab Size', color='#34495E', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, unique_tokens, width, label='Tokens Used', color=colors, alpha=0.9)

ax1.set_ylabel('Count', fontsize=11, weight='bold')
ax1.set_title('Tokenizer Vocabulary: Size vs Usage', fontsize=12, weight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(languages, fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)

# Chart 2: Chars per Token (Fertility)
chars_per_token = [1.3327, 5.9283, 5.8353, 1.5412, 3.4369, 3.5]  # approximate for WP
bars = ax2.barh(languages, chars_per_token, color=colors, alpha=0.8)
ax2.set_xlabel('Average Characters per Token', fontsize=11, weight='bold')
ax2.set_title('Tokenization Fertility Analysis', fontsize=12, weight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, chars_per_token)):
    ax2.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10, weight='bold')

# Chart 3: UNK Rate Comparison
unk_rates = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001]
bars = ax3.bar(x_pos, unk_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Unknown Token Rate (%)', fontsize=11, weight='bold')
ax3.set_title('Tokenizer Coverage: Unknown Token Rate', fontsize=12, weight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(languages, fontsize=9)
ax3.set_ylim(0, 0.00015)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, unk_rates):
    if val > 0:
        ax3.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}%',
                ha='center', va='bottom', fontsize=9, weight='bold')
    else:
        ax3.text(bar.get_x() + bar.get_width()/2., 0.00001,
                '0.0%',
                ha='center', va='bottom', fontsize=9, weight='bold')

# Chart 4: Tokenizer Type Summary
ax4.axis('off')

summary_text = """
TOKENIZER TRAINING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TELUGU (Model H) - 3 Variants × 50K Vocabulary
──────────────────────────────────────────────────
  Byte-level BPE:     1,978 tokens used | 1.33 chars/token
  Unicode-level BPE: 11,398 tokens used | 5.93 chars/token
  WordPiece:          9,539 tokens used | 5.84 chars/token
  ✓ All 0.0% UNK rate (100% corpus coverage)

BHOJPURI (Model L) - 3 Variants
──────────────────────────────────────────────────
  Byte-level BPE (8K):     7,761 tokens | 1.54 chars/token
  Unicode-level BPE (16K): 15,643 tokens | 3.44 chars/token
  WordPiece (16K):         ~15,000 tokens | ~3.5 chars/token
  ✓ 0.0000-0.0001% UNK rate (~100% coverage)

KEY METRICS
──────────────────────────────────────────────────
  • Total Tokenizers Trained: 6
  • Vocabulary Sizes: 8K, 16K (Bhojpuri), 50K (Telugu)
  • Average Fertility: 1.3-5.9 chars/token
  • Coverage: 100% (near-zero UNK rates)
  • Training Data: Telugu test 2.5M lines, Bhojpuri 77.7K lines
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.9, pad=1))

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_tokenizer_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_tokenizer_analysis.png")
plt.close()

print("✅ Tokenizer training visualization created!")
