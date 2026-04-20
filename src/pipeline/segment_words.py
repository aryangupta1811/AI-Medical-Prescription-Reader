import cv2
import numpy as np
import os

def segment_words(image_path, output_dir="segments"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # dilation to group characters into words
    kernel = np.ones((3, 15), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # filter noise
        if w < 30 or h < 15:
            continue

        word = gray[y:y+h, x:x+w]

        filename = os.path.join(output_dir, f"word_{i}.png")
        cv2.imwrite(filename, word)

        segments.append(filename)

    return segments