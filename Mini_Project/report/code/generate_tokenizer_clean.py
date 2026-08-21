import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Labels for all tokenizers
languages = ['Telugu\nByte', 'Telugu\nUnicode', 'Telugu\nWP',
             'Bhojpuri\nByte', 'Bhojpuri\nUnicode', 'Bhojpuri\nWP']
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']

# Chart 1: Chars per Token (Fertility)
chars_per_token = [1.3327, 5.9283, 5.8353, 1.5412, 3.4369, 3.5]
bars = ax1.barh(languages, chars_per_token, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Average Characters per Token', fontsize=12, weight='bold')
ax1.set_title('Tokenization Fertility', fontsize=13, weight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, chars_per_token)):
    ax1.text(val + 0.15, i, f'{val:.2f}', va='center', fontsize=11, weight='bold')

# Chart 2: UNK Rate Comparison
x_pos = np.arange(len(languages))
unk_rates = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001]
bars = ax2.bar(x_pos, unk_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Unknown Token Rate (%)', fontsize=12, weight='bold')
ax2.set_title('Corpus Coverage: UNK Rate', fontsize=13, weight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(languages, fontsize=10)
ax2.set_ylim(0, 0.00015)
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, unk_rates):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}%',
                ha='center', va='bottom', fontsize=10, weight='bold')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., 0.00001,
                '0.0%',
                ha='center', va='bottom', fontsize=10, weight='bold')

# Chart 3: Test Set Tokens (log scale)
test_tokens = [98469, 22136, 22489, 13613342, 6072593, 5930926]
bars = ax3.bar(x_pos, test_tokens, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Test Tokens Evaluated', fontsize=12, weight='bold')
ax3.set_title('Test Set Tokens Processed', fontsize=13, weight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(languages, fontsize=10)
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3, which='both')

for bar, val in zip(bars, test_tokens):
    label = f'{val/1e6:.1f}M' if val >= 1e6 else f'{val/1e3:.0f}K'
    ax3.text(bar.get_x() + bar.get_width()/2., val * 1.5,
            label,
            ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_tokenizer_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_tokenizer_analysis.png (3 plots only, no vocab size chart)")
plt.close()

print("✅ Tokenizer analysis chart updated - removed vocabulary size plot!")
