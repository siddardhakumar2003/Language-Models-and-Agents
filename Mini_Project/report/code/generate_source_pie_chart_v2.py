import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['font.family'] = 'sans-serif'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Telugu Data Sources
telugu_labels = ['Manual Corpus\n(te.txt)\n21.5M lines',
                 'Web Scraping\n29.2K lines',
                 'OCR Augmentation\n3.8M lines']
telugu_sizes = [21458261, 29206, 3766703]
colors_telugu = ['#2ECC71', '#3498DB', '#E74C3C']

# Create wedges with better explosion
wedges1, texts1, autotexts1 = ax1.pie(
    telugu_sizes,
    labels=telugu_labels,
    autopct='%1.1f%%',
    colors=colors_telugu,
    startangle=90,
    explode=(0.05, 0.1, 0.05),
    textprops={'fontsize': 10, 'weight': 'bold'},
    pctdistance=0.85
)

for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

ax1.set_title('Telugu (Model H) - Data Source Composition\nTotal: 25.3M lines | 4.4B tokens (884.4% of 500M target)',
              fontsize=12, weight='bold', pad=20)

# Bhojpuri Data Sources - with better positioning
bhojpuri_labels = [
    'HuggingFace Corpus\n386K docs',
    'OCR Augmentation\n511.6K lines',
    'Web Scraping\n3.2K lines',
    'MT Translation\n15.8K lines'
]
bhojpuri_sizes = [386000, 511599, 3234, 15852]
colors_bhojpuri = ['#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']

wedges2, texts2, autotexts2 = ax2.pie(
    bhojpuri_sizes,
    labels=bhojpuri_labels,
    autopct='%1.1f%%',
    colors=colors_bhojpuri,
    startangle=90,
    explode=(0.08, 0.08, 0.12, 0.12),
    textprops={'fontsize': 10, 'weight': 'bold'},
    pctdistance=0.85
)

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

ax2.set_title('Bhojpuri (Model L) - Data Source Composition\nTotal: 776.8K lines | 135.4M tokens (27.1% of 500M target)',
              fontsize=12, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_data_sources_both.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_data_sources_both.png (high quality, no overlaps)")
plt.close()

# Create Token Distribution Pie Chart with better layout
fig, ax = plt.subplots(figsize=(11, 8))

languages = ['Telugu\n(4.4B tokens)', 'Bhojpuri\n(135.4M tokens)']
token_counts = [4421917647, 135361900]
colors_progress = ['#27AE60', '#E74C3C']

wedges, texts, autotexts = ax.pie(
    token_counts,
    labels=languages,
    autopct=lambda pct: f'{pct:.1f}%',
    colors=colors_progress,
    startangle=90,
    explode=(0.08, 0.08),
    textprops={'fontsize': 12, 'weight': 'bold'},
    pctdistance=0.85,
    wedgeprops=dict(width=0.8, edgecolor='white', linewidth=2)
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_weight('bold')

# Add center text (donut style)
centre_circle = plt.Circle((0, 0), 0.70, fc='white', edgecolor='white', linewidth=2)
ax.add_artist(centre_circle)

# Add legend with detailed stats
legend_text = [
    f'Telugu: 4,421,917,647 tokens\n(884.4% of 500M target) ✓ EXCEEDED',
    f'Bhojpuri: 135,361,900 tokens\n(27.1% of 500M target) - Phase 3 expansion ongoing'
]
ax.legend(legend_text, loc='center', fontsize=11, frameon=True,
          fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.05))

ax.set_title('Token Distribution Across Languages\nProgress Toward 500M Token Target per Language',
             fontsize=14, weight='bold', pad=30)

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_token_distribution.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_token_distribution.png (donut chart with center legend)")
plt.close()

# Create combined stats visualization
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top: Data Collection Overview
ax1 = fig.add_subplot(gs[0, :])
categories = ['Telugu', 'Bhojpuri']
manual = [21458261, 0]
web = [29206, 3234]
ocr = [3766703, 511599]
mt = [0, 15852]
hf = [0, 386000]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - 1.8*width, manual, width, label='Manual Corpus', color='#2ECC71')
bars2 = ax1.bar(x - 0.9*width, web, width, label='Web Scraping', color='#3498DB')
bars3 = ax1.bar(x, ocr, width, label='OCR Augmentation', color='#E74C3C')
bars4 = ax1.bar(x + 0.9*width, mt, width, label='MT Translation', color='#F39C12')
bars5 = ax1.bar(x + 1.8*width, hf, width, label='HuggingFace Corpus', color='#9B59B6')

ax1.set_ylabel('Number of Lines', fontsize=12, weight='bold')
ax1.set_title('Data Collection Breakdown by Source and Language', fontsize=13, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11, weight='bold')
ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=8, rotation=0)

# Bottom Left: Language Stats
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
stats_text = """
TELUGU (MODEL H) - HIGHER-RESOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Lines: 25,349,199
Total Tokens: 4,421,917,647 (4.4B)
Progress: 884.4% ✓ EXCEEDED
Train/Val/Test: 20.3M / 2.5M / 2.5M
Corpus Size: 17.4 GB

BHOJPURI (MODEL L) - LOWER-RESOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Lines: 776,810
Total Tokens: 135,361,900 (135.4M)
Progress: 27.1% (Phase 3: Ongoing)
Train/Val/Test: 621.8K / 77.4K / 77.7K
Corpus Size: 516.4 MB
"""
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

# Bottom Right: Tokenizer Summary
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
tokenizer_text = """
TOKENIZER TRAINING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Telugu (3 variants × 50K vocab):
  • Byte-level: 1,978 tokens used
  • Unicode-level: 11,398 tokens used
  • WordPiece: 9,539 tokens used

Bhojpuri (3 variants):
  • Byte-level (8K): 7,761 tokens used
  • Unicode-level (16K): 15,643 tokens
  • WordPiece (16K): 5.9M test tokens

All tokenizers: 0.0000-0.0001% UNK rate
(100% corpus coverage)
"""
ax3.text(0.05, 0.95, tokenizer_text, transform=ax3.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.8))

plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_phase1_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_phase1_summary.png (comprehensive overview)")
plt.close()

print("\n✅ All charts regenerated with improved layout and no overlaps!")
