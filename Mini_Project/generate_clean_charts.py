import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CHART 1: Tokenizer Analysis (4 panels - NO TEXT BOX)
# ============================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

# Chart 1: Vocabulary Size Comparison
languages = ['Telugu\nByte', 'Telugu\nUnicode', 'Telugu\nWP',
             'Bhojpuri\nByte', 'Bhojpuri\nUnicode', 'Bhojpuri\nWP']
vocab_sizes = [50000, 50000, 50000, 8000, 16000, 16000]
unique_tokens = [1978, 11398, 9539, 7761, 15643, 15000]

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']

x_pos = np.arange(len(languages))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, vocab_sizes, width, label='Vocab Size', color='#34495E', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, unique_tokens, width, label='Tokens Used', color=colors, alpha=0.9)

ax1.set_ylabel('Count', fontsize=11, weight='bold')
ax1.set_title('Vocabulary Size vs Usage', fontsize=12, weight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(languages, fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)

# Chart 2: Chars per Token (Fertility)
chars_per_token = [1.3327, 5.9283, 5.8353, 1.5412, 3.4369, 3.5]
bars = ax2.barh(languages, chars_per_token, color=colors, alpha=0.8)
ax2.set_xlabel('Average Characters per Token', fontsize=11, weight='bold')
ax2.set_title('Tokenization Fertility', fontsize=12, weight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, chars_per_token)):
    ax2.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10, weight='bold')

# Chart 3: UNK Rate Comparison
unk_rates = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001]
bars = ax3.bar(x_pos, unk_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Unknown Token Rate (%)', fontsize=11, weight='bold')
ax3.set_title('Corpus Coverage: UNK Rate', fontsize=12, weight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(languages, fontsize=9)
ax3.set_ylim(0, 0.00015)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, unk_rates):
    if val > 0:
        ax3.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}%',
                ha='center', va='bottom', fontsize=9, weight='bold')
    else:
        ax3.text(bar.get_x() + bar.get_width()/2., 0.00001,
                '0.0%',
                ha='center', va='bottom', fontsize=9, weight='bold')

# Chart 4: Test Set Tokens (removed text box, added bar chart instead)
test_tokens = [98469, 22136, 22489, 13613342, 6072593, 5930926]
bars = ax4.bar(x_pos, test_tokens, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Test Tokens Evaluated', fontsize=11, weight='bold')
ax4.set_title('Test Set Tokens Processed', fontsize=12, weight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(languages, fontsize=9)
ax4.set_yscale('log')
ax4.grid(axis='y', alpha=0.3, which='both')

for bar, val in zip(bars, test_tokens):
    ax4.text(bar.get_x() + bar.get_width()/2., val * 1.5,
            f'{val/1e6:.1f}M' if val >= 1e6 else f'{val/1e3:.0f}K',
            ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_tokenizer_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_tokenizer_analysis.png (plot only, no text box)")
plt.close()

# ============================================================
# CHART 2: Phase 1 Summary (3 plots - NO TEXT BOXES)
# ============================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top: Data Collection Overview (log scale)
ax1 = fig.add_subplot(gs[0, :])
categories = ['Telugu', 'Bhojpuri']
manual = [21458261, 0]
web = [29206, 3234]
ocr = [3766703, 511599]
mt = [0, 15852]
hf = [0, 386000]

x = np.arange(len(categories))
width = 0.15

bars1 = ax1.bar(x - 2*width, manual, width, label='Manual Corpus', color='#2ECC71')
bars2 = ax1.bar(x - width, web, width, label='Web Scraping', color='#3498DB')
bars3 = ax1.bar(x, ocr, width, label='OCR Augmentation', color='#E74C3C')
bars4 = ax1.bar(x + width, mt, width, label='MT Translation', color='#F39C12')
bars5 = ax1.bar(x + 2*width, hf, width, label='HuggingFace Corpus', color='#9B59B6')

ax1.set_ylabel('Number of Lines', fontsize=12, weight='bold')
ax1.set_title('Data Collection Breakdown by Source and Language', fontsize=13, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11, weight='bold')
ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, which='both')

for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height * 2,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=8, rotation=0)

# Bottom Left: Data Statistics Bar Chart
ax2 = fig.add_subplot(gs[1, 0])
languages_full = ['Telugu\nLines', 'Telugu\nTokens\n(B)', 'Bhojpuri\nLines', 'Bhojpuri\nTokens\n(M)']
data_values = [25349199/1e6, 4.4, 776810/1e3, 135.4]
colors_data = ['#2ECC71', '#27AE60', '#E74C3C', '#C0392B']
bars = ax2.bar(languages_full, data_values, color=colors_data, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Value', fontsize=11, weight='bold')
ax2.set_title('Dataset Scale Comparison', fontsize=12, weight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, data_values):
    ax2.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=10, weight='bold')

# Bottom Right: Progress Toward Target
ax3 = fig.add_subplot(gs[1, 1])
languages_prog = ['Telugu', 'Bhojpuri']
progress = [884.4, 27.1]
colors_prog = ['#27AE60', '#E74C3C']
bars = ax3.bar(languages_prog, progress, color=colors_prog, alpha=0.8, edgecolor='black', linewidth=2)
ax3.axhline(y=100, color='black', linestyle='--', linewidth=2, label='500M Target (100%)')
ax3.set_ylabel('Progress (%)', fontsize=11, weight='bold')
ax3.set_title('Token Progress Toward 500M Target', fontsize=12, weight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.legend(fontsize=10)

for bar, val in zip(bars, progress):
    ax3.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, weight='bold')

plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_phase1_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_phase1_summary.png (plots only, no text boxes)")
plt.close()

print("\n✅ All charts regenerated without description boxes!")
