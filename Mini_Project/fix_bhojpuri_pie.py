import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Telugu Data Sources
telugu_labels = ['Manual Corpus\n(te.txt)\n21.5M lines',
                 'Web Scraping\n29.2K lines',
                 'OCR Augmentation\n3.8M lines']
telugu_sizes = [21458261, 29206, 3766703]
colors_telugu = ['#2ECC71', '#3498DB', '#E74C3C']

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

# Bhojpuri Data Sources - with BETTER LABEL POSITIONING
bhojpuri_labels = ['HuggingFace\nCorpus',
                   'OCR\nAugmentation',
                   'Web\nScraping',
                   'Machine\nTranslation']
bhojpuri_sizes = [386000, 511599, 3234, 15852]
colors_bhojpuri = ['#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']

wedges2, texts2, autotexts2 = ax2.pie(
    bhojpuri_sizes,
    labels=bhojpuri_labels,
    autopct='%1.1f%%',
    colors=colors_bhojpuri,
    startangle=45,  # Changed starting angle for better distribution
    explode=(0.1, 0.1, 0.15, 0.15),  # Increased explosion for spacing
    textprops={'fontsize': 10, 'weight': 'bold'},
    pctdistance=0.80
)

# Manually adjust text positions for better clarity
for text in texts2:
    text.set_fontsize(10)
    text.set_weight('bold')
    # Adjust radial distance of labels
    x, y = text.get_position()
    length = np.sqrt(x**2 + y**2)
    if length > 0:
        text.set_position((1.35*x/length, 1.35*y/length))

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')

ax2.set_title('Bhojpuri (Model L) - Data Source Composition\nTotal: 776.8K lines | 135.4M tokens (27.1% of 500M target)',
              fontsize=12, weight='bold', pad=20)

# Add legend for Bhojpuri with full names
legend_labels = [
    'HuggingFace Corpus: 386K docs',
    'OCR Augmentation: 511.6K lines',
    'Web Scraping: 3.2K lines',
    'Machine Translation: 15.8K lines'
]
ax2.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=9, frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_data_sources_both.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Fixed: plot_data_sources_both.png - Bhojpuri labels repositioned with legend")
plt.close()

print("✅ Bhojpuri pie chart labels corrected - no more overlaps!")
