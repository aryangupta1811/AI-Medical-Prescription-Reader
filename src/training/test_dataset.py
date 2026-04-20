from src.training.dataset import OCRDataset, collate_fn
from torch.utils.data import DataLoader

dataset = OCRDataset(
    labels_path="data/processed/labels_balanced.txt",
    images_dir="data/processed/images_balanced"
)

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for batch in loader:
    images, labels, label_lengths = batch

    print("Images shape:", images.shape)
    print("Labels:", labels)
    print("Label lengths:", label_lengths)
    break