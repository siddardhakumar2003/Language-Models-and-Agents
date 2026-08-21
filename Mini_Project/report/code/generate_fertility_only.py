import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))

# Labels for all tokenizers
languages = ['Telugu\nByte-level', 'Telugu\nUnicode-level', 'Telugu\nWordPiece',
             'Bhojpuri\nByte-level', 'Bhojpuri\nUnicode-level', 'Bhojpuri\nWordPiece']
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']

# Fertility (Chars per Token)
chars_per_token = [1.3327, 5.9283, 5.8353, 1.5412, 3.4369, 3.5]

bars = ax.barh(languages, chars_per_token, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Average Characters per Token', fontsize=13, weight='bold')
ax.set_title('Tokenizer Fertility Analysis: Compression Efficiency', fontsize=14, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 7)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, chars_per_token)):
    ax.text(val + 0.15, i, f'{val:.2f}', va='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/report/phase-1/plot_tokenizer_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: plot_tokenizer_analysis.png (fertility rate only)")
plt.close()

print("✅ Tokenizer analysis chart updated - showing only fertility rate!")
