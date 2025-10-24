import torch
from model import TinyTransformer
def summarize_checkpoint(path="checkpoint.pt"):
    ckpt = torch.load(path, map_location="cpu")
    print(f"Checkpoint summary for: {path}")
    print(f"  Epoch: {ckpt['epoch']}")        
    print(f"  Vocab size: {len(ckpt['vocab']['stoi'])}")
    print("\nSample vocab entries:", list(ckpt['vocab']['stoi'].items())[:10])

    model = TinyTransformer(len(ckpt['vocab']['stoi']))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    for name, param in model.named_parameters():
        print(f"{name:30s} {param.numel():10d}")


summarize_checkpoint()
