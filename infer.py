import torch
from model import TinyTransformer

ckpt_path = "checkpoint.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu')
stoi = checkpoint['vocab']['stoi']
itos = checkpoint['vocab']['itos']
vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = TinyTransformer(vocab_size)
model.load_state_dict(checkpoint['model'])
model.eval()

def generate(model, start="திரு", max_new_tokens=200, block_size=128):
    device = next(model.parameters()).device
    idx = torch.tensor([encode(start)], dtype=torch.long)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return decode(idx[0].tolist())

print(generate(model, "திரு"))
