import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors_telugu = ['#FF6B6B', '#4ECDC4', '#45B7D1']
colors_bhojpuri = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Telugu Data Sources
telugu_labels = ['Manual Corpus\n(te.txt)', 'Web Scraping', 'OCR Augmentation']
telugu_sizes = [21458261, 29206, 3766703]
telugu_percentages = [84.6, 0.1, 14.9]

wedges1, texts1, autotexts1 = ax1.pie(
    telugu_sizes,
    labels=telugu_labels,
    autopct='%1.1f%%',
    colors=colors_telugu,
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

ax1.set_title('Telugu (Model H) - Data Source Composition\nTotal: 25.3M lines (4.4B tokens)',
              fontsize=13, weight='bold', pad=20)

# Bhojpuri Data Sources
bhojpuri_labels = [
    'HuggingFace Corpus\n(386K docs)',
    'OCR Augmentation\n(511.6K lines)',
    'Web Scraping\n(3.2K lines)',
    'MT Translation\n(15.8K lines)'
]
bhojpuri_sizes = [386000, 511599, 3234, 15852]
bhojpuri_percentages = [49.7, 65.9, 0.4, 2.0]

wedges2, texts2, autotexts2 = ax2.pie(
    bhojpuri_sizes,
    labels=bhojpuri_labels,
    autopct='%1.1f%%',
    colors=colors_bhojpuri,
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

ax2.set_title('Bhojpuri (Model L) - Data Source Composition\nTotal: 776.8K lines (135.4M tokens)',
              fontsize=13, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_data_sources_both.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: plot_data_sources_both.png")
plt.close()

# Create alternative: Token Progress Pie Chart
fig, ax = plt.subplots(figsize=(10, 7))

languages = ['Telugu\n(4.4B tokens)', 'Bhojpuri\n(135.4M tokens)']
token_progress = [4421917647, 135361900]
target = 500000000

# Create pie showing progress toward 500M target
colors_progress = ['#2ECC71', '#E74C3C']
sizes = [min(tok, target) for tok in token_progress]
explode = (0.05, 0.05)

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=languages,
    autopct=lambda pct: f'{pct:.1f}%',
    colors=colors_progress,
    startangle=90,
    explode=explode,
    textprops={'fontsize': 12, 'weight': 'bold'}
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

ax.set_title('Token Distribution Across Languages\n(500M Token Target)',
             fontsize=14, weight='bold', pad=20)

# Add legend with token counts
legend_labels = [
    f'Telugu: {token_progress[0]:,} tokens (884.4% of target)',
    f'Bhojpuri: {token_progress[1]:,} tokens (27.1% of target)'
]
ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.85, 0.15), fontsize=11)

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_token_distribution.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: plot_token_distribution.png")
plt.close()

print("\n✅ All pie charts generated successfully!")
