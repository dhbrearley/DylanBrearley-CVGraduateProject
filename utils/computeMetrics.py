import cv2
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from skimage.filters.rank import entropy
from skimage.morphology import disk

def compute_metrics(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness (via Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Glare ratio (bright pixel %)
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

    # 7. Saturation level (from HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].mean()

    return sharpness, glare_ratio, contrast, ent, edge_density, mean_brightness, saturation

data = []
for f in tqdm(glob.glob("../Frames/*.jpg")):
    img = cv2.imread(f)
    metrics = compute_metrics(img)
    data.append([f] + list(metrics))

df = pd.DataFrame(data, columns=[
    "frame", "sharpness", "glare", "contrast", 
    "entropy", "edge_density", "mean_brightness", "saturation"
])
df.to_csv("../Data/metrics.csv", index=False)

