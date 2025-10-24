import torch, os
from model import TinyTransformer
from torch.nn import functional as F

# Load data
with open("data/thirukkural.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
text = "\n".join(lines)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)

# Dataset splits
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
block_size, batch_size = 128, 16
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Initialize or load checkpoint
model = TinyTransformer(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

start_epoch = 0

ckpt_path = "checkpoint.pt"
if os.path.exists(ckpt_path):
    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1

# Training
epochs = 10
for epoch in range(start_epoch, epochs):
    model.train()
    losses = []
    for step in range(500):  # steps per epoch
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step}: loss {loss.item():.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vocab': {'stoi': stoi, 'itos': itos}
    }, ckpt_path)
    print(f"Checkpoint saved at epoch {epoch}")
