# eval_perplexity.py
import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def make_toy_texts(num=1024):
    base = [
        "The quick brown fox jumps over the lazy dog.",
        "Large language models enable powerful text generation.",
        "Parameter-efficient fine-tuning reduces compute and memory.",
        "Activation memory dominates backpropagation costs.",
        "LoRA adds low-rank adapters to attention projections.",
        "Tiny datasets are handy for quick debugging.",
        "Edge devices benefit from compact models.",
    ]
    texts = (base * ((num + len(base) - 1) // len(base)))[:num]
    return texts

def evaluate(model, tokenizer, texts, device="cuda"):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            batch_texts = texts[i:i+8]
            enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            for k in enc: enc[k] = enc[k].to(device)
            enc["labels"] = enc["input_ids"].clone()
            out = model(**enc)
            losses.append(out.loss.item())
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    print(f"Mean loss: {mean_loss:.4f} | Perplexity: {perplexity:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory with model and tokenizer")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    texts = make_toy_texts(1024)
    evaluate(model, tokenizer, texts, device)

if __name__ == "__main__":
    main()
