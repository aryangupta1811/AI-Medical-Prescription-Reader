from collections import Counter
import os
import random
import cv2

LABELS_PATH = "data/processed/labels_balanced.txt"
IMAGES_PATH = "data/processed/images_balanced/"

def load_labels():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"❌ labels file not found: {LABELS_PATH}")
    
    lines = []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # skip malformed
            lines.append((parts[0], parts[1]))
    
    if len(lines) == 0:
        raise RuntimeError("❌ labels_balanced.txt is EMPTY")

    return lines


def analyze_distribution(lines):
    labels = [label for _, label in lines]
    counter = Counter(labels)

    print("\n==============================")
    print("📊 DATASET DISTRIBUTION")
    print("==============================")
    print("Total samples:", len(labels))
    print("Unique labels:", len(counter))

    print("\nTop 10 labels:")
    for k, v in counter.most_common(10):
        print(f"{k}: {v}")

    print("\nMin samples per label:", min(counter.values()))
    print("Max samples per label:", max(counter.values()))

    # 🚨 strict checks
    if max(counter.values()) > 20:
        print("\n❌ ERROR: Some labels exceed max limit (20)")
    else:
        print("\n✅ All labels within limit")

    if len(counter) < 50:
        print("⚠️ WARNING: Too few unique labels (dataset still weak)")


def check_missing_images(lines):
    print("\n==============================")
    print("🖼️ IMAGE CONSISTENCY CHECK")
    print("==============================")

    missing = 0
    for fname, _ in lines:
        path = os.path.join(IMAGES_PATH, fname)
        if not os.path.exists(path):
            missing += 1

    print("Missing images:", missing)

    if missing > 0:
        raise RuntimeError("❌ Some images are missing. Fix dataset before training.")
    else:
        print("✅ All images present")


def visual_sanity_check(lines, num_samples=10):
    print("\n==============================")
    print("👁️ VISUAL SANITY CHECK")
    print("==============================")

    samples = random.sample(lines, min(num_samples, len(lines)))

    for fname, label in samples:
        path = os.path.join(IMAGES_PATH, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Failed to load: {fname}")
            continue

        print(f"Image: {fname} | Label: {label}")

        # show image
        cv2.imshow("Sample", img)
        cv2.waitKey(500)  # 0.5 sec per image

    cv2.destroyAllWindows()
    print("✅ Visual check done")


def main():
    print("\n🚀 VALIDATING BALANCED DATASET...\n")

    lines = load_labels()
    analyze_distribution(lines)
    check_missing_images(lines)
    visual_sanity_check(lines)

    print("\n✅ DATASET VALIDATION COMPLETE")


if __name__ == "__main__":
    main()