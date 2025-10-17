import cv2
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from skimage.filters.rank import entropy
from skimage.morphology import disk

# I tested each of these on PIV to see which ranges would cause a problem when measuring
MIN_BLUR = 41
MAX_BLUR = 121
MIN_LIGHTNESS = 70
MAX_LIGHTNESS = 110
MIN_DARKNESS = 70
MAX_DARKNESS = 120

def compute_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    glare_ratio = np.mean(gray > 240)
    contrast = gray.std()
    gray_uint8 = gray.astype(np.uint8)
    ent = entropy(gray_uint8, disk(5)).mean()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    mean_brightness = gray.mean()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].mean()
    return sharpness, glare_ratio, contrast, ent, edge_density, mean_brightness, saturation

def generate_bad_frame(frame):
    altered = frame.copy()
    reasons = []
    applied = False

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


    if np.random.rand() < 0.3:
        ksize = np.random.choice(range(MIN_BLUR, MAX_BLUR+1, 2))  # force kernel odd
        altered = cv2.GaussianBlur(altered, (ksize, ksize), 0)
        reasons.append("motion_blur")
        applied = True

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
            ksize = np.random.choice(range(MIN_BLUR, MAX_BLUR+1, 2))
            altered = cv2.GaussianBlur(altered, (ksize, ksize), 0)
            reasons.append("motion_blur")

    return altered, ", ".join(reasons)

def process_videos(input_folder, output_folder, csv_path, existing_metrics_csv=None):
    os.makedirs(output_folder, exist_ok=True)
    data = []

    if existing_metrics_csv and os.path.exists(existing_metrics_csv):
        df_existing = pd.read_csv(existing_metrics_csv)
        data.extend(df_existing.to_numpy().tolist())

    for f in tqdm(glob.glob(os.path.join(input_folder, "*.jpg"))):
        frame = cv2.imread(f)
        bad_frame, reason = generate_bad_frame(frame)
        filename = os.path.join(output_folder, os.path.basename(f))
        cv2.imwrite(filename, bad_frame)

        metrics = compute_metrics(bad_frame)
        data.append([filename, *metrics, "bad", reason])

    columns = [
        "frame", "sharpness", "glare", "contrast",
        "entropy", "edge_density", "mean_brightness", "saturation",
        "label", "reason"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

def main():
    input_folder = "../Frames"
    output_folder = "../Frames_Bad"
    metrics_csv = "../Data/metrics_labeled.csv"

    process_videos(input_folder, output_folder, metrics_csv)

if __name__ == "__main__":
    main()
