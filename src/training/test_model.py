import torch
from src.training.model import CRNN

# same charset as dataset
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(CHARS) + 1

model = CRNN(num_classes)

x = torch.randn(4, 1, 32, 128)  # batch of 4

out = model(x)

print("Output shape:", out.shape)