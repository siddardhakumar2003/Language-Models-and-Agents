#!/usr/bin/env python3
"""Generate actual samples from trained models for report."""

import torch
import json
import sys
sys.path.insert(0, '/media/ubuntu/Personal/IIIT_Hyderabad/Semester_3/LMA/Mini_Project')

from telugu.model.transformer import TeluguTransformer
from bhojpuri.model.transformer import BhojpuriTransformer
from telugu.tokenizer.tokenizer_wrapper import TeluguTokenizer
from bhojpuri.tokenizer.tokenizer_wrapper import BhojpuriTokenizer

def load_model_and_tokenizer(model_type):
    """Load model and tokenizer from checkpoint."""
    if model_type == "telugu":
        model = TeluguTransformer()
        tokenizer = TeluguTokenizer()
        checkpoint = torch.load("telugu/model/outputs/checkpoints/checkpoint_best.pt",
                               map_location=torch.device('cpu'))
    else:
        model = BhojpuriTransformer()
        tokenizer = BhojpuriTokenizer()
        checkpoint = torch.load("bhojpuri/model/outputs/checkpoints/checkpoint_best.pt",
                               map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer

def generate_samples(model_type, num_samples=3, max_new_tokens=15):
    """Generate text samples from model."""
    model, tokenizer = load_model_and_tokenizer(model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            # Random prompt
            prompt_tokens = torch.randint(100, 500, (1, 3)).to(device)
            generated = model.generate(prompt_tokens, max_new_tokens=max_new_tokens,
                                      temperature=1.0, greedy=False)

            # Decode
            tokens_list = generated[0].tolist()
            text = tokenizer.decode(tokens_list)
            samples.append({
                "tokens": tokens_list,
                "text": text,
                "num_tokens": len(tokens_list)
            })

    return samples

if __name__ == "__main__":
    # Generate samples
    telugu_samples = generate_samples("telugu", num_samples=3)
    bhojpuri_samples = generate_samples("bhojpuri", num_samples=3)

    # Save to JSON
    output = {
        "telugu": telugu_samples,
        "bhojpuri": bhojpuri_samples
    }

    with open("report/phase-2/generated_samples.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print for review
    print("Telugu Samples:")
    for i, sample in enumerate(telugu_samples):
        print(f"{i+1}. {sample['text']}")
        print(f"   Tokens ({len(sample['tokens'])}): {sample['tokens']}\n")

    print("\nBhojpuri Samples:")
    for i, sample in enumerate(bhojpuri_samples):
        print(f"{i+1}. {sample['text']}")
        print(f"   Tokens ({len(sample['tokens'])}): {sample['tokens']}\n")
