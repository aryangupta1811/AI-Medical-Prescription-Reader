import cv2
import numpy as np
import os


def segment_lines(image_path, output_dir="line_segments"):

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 adaptive threshold (better for uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    # 🔥 horizontal dilation → merge words into lines
    kernel = np.ones((5, 100), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # find contours (lines)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    # sort top to bottom
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # filter noise
        if h < 20 or w < 100:
            continue

        line_img = gray[y:y+h, x:x+w]

        filename = os.path.join(output_dir, f"line_{i}.png")
        cv2.imwrite(filename, line_img)

        lines.append(filename)

    return lines