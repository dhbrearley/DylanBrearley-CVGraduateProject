import cv2
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from skimage.filters.rank import entropy
from skimage.morphology import disk

INPUT_FOLDER = "../Frames"
BAD_OUTPUT_FOLDER = "../Frames_Bad"
DATA_DIR = "../Data"
OUTPUT_CSV = os.path.join(DATA_DIR, "metrics_labeled.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BAD_OUTPUT_FOLDER, exist_ok=True)

MIN_BLUR = 41
MAX_BLUR = 121
MIN_LIGHTNESS = 70
MAX_LIGHTNESS = 110
MIN_DARKNESS = 70
MAX_DARKNESS = 120

def compute_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Glare ratio (bright pixels)
    glare_ratio = np.mean(gray > 240)

    # 3. Contrast (std deviation)
    contrast = gray.std()

    # 4. Entropy (texture complexity)
    gray_uint8 = gray.astype(np.uint8)
    ent = entropy(gray_uint8, disk(5)).mean()

    # 5. Edge density (Canny edges / total pixels)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    # 6. Mean brightness
    mean_brightness = gray.mean()

    # 7. Saturation level
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()

    # 8. Clipped black ratio (underexposure)
    clipped_black_ratio = np.mean(gray < 10)

    # 9. Clipped white ratio (overexposure)
    clipped_white_ratio = np.mean(gray > 245)

    return (
        sharpness, glare_ratio, contrast, ent, edge_density,
        mean_brightness, saturation, clipped_black_ratio, clipped_white_ratio
    )

def generate_bad_frame(frame):
    altered = frame.copy()
    reasons = []
    applied = False

    # Random darkening or brightening
    if np.random.rand() < 0.5:
        beta = np.random.randint(MIN_DARKNESS, MAX_DARKNESS)
        altered = cv2.convertScaleAbs(altered, alpha=1.0, beta=-beta)
        reasons.append("too_dark")
        applied = True
    elif np.random.rand() < 0.5:
        beta = np.random.randint(MIN_LIGHTNESS, MAX_LIGHTNESS)
        altered = cv2.convertScaleAbs(altered, alpha=1.0, beta=beta)
        reasons.append("too_bright")
        applied = True

    # Random blur
    if np.random.rand() < 0.3:
        ksize = np.random.choice(range(MIN_BLUR, MAX_BLUR + 1, 2))
        altered = cv2.GaussianBlur(altered, (ksize, ksize), 0)
        reasons.append("motion_blur")
        applied = True

    # Ensure at least one alteration is applied
    if not applied:
        choice = np.random.choice(["darken", "lighten", "blur"])
        if choice == "darken":
            beta = np.random.randint(MIN_DARKNESS, MAX_DARKNESS)
            altered = cv2.convertScaleAbs(altered, alpha=1.0, beta=-beta)
            reasons.append("too_dark")
        elif choice == "lighten":
            beta = np.random.randint(MIN_LIGHTNESS, MAX_LIGHTNESS)
            altered = cv2.convertScaleAbs(altered, alpha=1.0, beta=beta)
            reasons.append("too_bright")
        else:
            ksize = np.random.choice(range(MIN_BLUR, MAX_BLUR + 1, 2))
            altered = cv2.GaussianBlur(altered, (ksize, ksize), 0)
            reasons.append("motion_blur")

    return altered, ", ".join(reasons)

def main():
    data = []

    print("Generating good...")
    for f in tqdm(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))):
        img = cv2.imread(f)
        if img is None:
            continue
        metrics = compute_metrics(img)
        data.append([f, *metrics, "good", "original"])

    print("\n⚙️ Generating bad...")
    for f in tqdm(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))):
        img = cv2.imread(f)
        if img is None:
            continue

        bad_frame, reason = generate_bad_frame(img)
        bad_filename = os.path.join(BAD_OUTPUT_FOLDER, os.path.basename(f))
        cv2.imwrite(bad_filename, bad_frame)

        metrics = compute_metrics(bad_frame)
        data.append([bad_filename, *metrics, "bad", reason])

    columns = [
        "frame", "sharpness", "glare", "contrast", "entropy", "edge_density",
        "mean_brightness", "saturation", "clipped_black_ratio",
        "clipped_white_ratio", "label", "reason"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinished")

if __name__ == "__main__":
    main()
