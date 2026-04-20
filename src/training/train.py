import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.dataset import OCRDataset, collate_fn
from src.training.model import CRNN

# =========================
# 🔹 CONFIGURATION
# =========================
BATCH_SIZE = 16
EPOCHS = 10        # Heavy constraints natively prevent CPU hardware lockout while rapidly tuning curve bounds
LR = 0.0001
VAL_SPLIT = 0.15   # 15% dedicated to Validation
PRETRAINED_MODEL = "model.pth" # Bootstraps mathematically perfectly from the clean pre-synthetic neural baseline

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(CHARS) + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 🔹 DATA & SPLIT
# =========================
full_dataset = OCRDataset(
    labels_path="data/processed/labels_balanced.txt",
    images_dir="data/processed/images_balanced"
)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

# Natively separate validation holdouts to guarantee integrity
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, drop_last=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, drop_last=False
)

# =========================
# 🔹 MODEL & TRANSFER LEARNING
# =========================
model = CRNN(num_classes).to(device)

if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
    print(f"\n[WAIT] Loading pre-trained weights from {PRETRAINED_MODEL}...")
    # Load strictly CNN and LSTM features, allowing output layer size mismatches
    model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device), strict=False)
    print("[OK] Transfer learning weights securely mapped!")

# =========================
# 🔹 OPTIMIZER + SCHEDULER
# =========================
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Radically reduces Learning Rate when the model plateaus, forcing deeper accuracy tuning
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"\n[START] Starting Robust Handwriting Training on {device}...")
print(f"Data Split: {train_size} Train | {val_size} Validation\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # ==========================
    # 🏃 TRAINING PHASE
    # ==========================
    model.train()
    total_train_loss = 0

    for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        outputs = outputs.permute(1, 0, 2)
        log_probs = nn.functional.log_softmax(outputs, dim=2)

        input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        
        # 🔥 CRITICAL: Prevents exploding gradients commonly seen in cursive CTC calculations
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) 
        
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ==========================
    # 🔍 VALIDATION PHASE
    # ==========================
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels, label_lengths in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            log_probs = nn.functional.log_softmax(outputs, dim=2)

            input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Step the learning rate dynamically based on Validation loss stalling
    scheduler.step(avg_val_loss)

    # ==========================
    # 🌟 BEST CHECKPOINTING
    # ==========================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        # Secures the BEST version statically specifically bypassing the corrupted previous runs
        torch.save(model.state_dict(), "model_augmented.pth")
        print(f"  [BEST] New Augmented OCR weights exported! (Val Loss: {best_val_loss:.4f})")

print("\n[OK] Validated Training Ecosystem Finished. Best active weights natively loaded to 'model_augmented.pth'.")
