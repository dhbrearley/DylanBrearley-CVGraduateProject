import cv2
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from skimage import exposure
from scipy.stats import entropy

VIDEO_PATH = "../Videos/ACS.MP4"
MODEL_PATH = "../Models/xgb_quality_model2.json"
SCALER_PATH = "../Models/quality_scaler2.joblib"
FRAME_INTERVAL = 10
REJECTION_THRESHOLD = 0.5

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def extract_features(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    glare = np.mean(gray > 240)

    contrast = gray.std()

    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    ent = entropy(hist + 1e-7)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    mean_brightness = gray.mean()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()

    return [sharpness, glare, contrast, ent, edge_density, mean_brightness, saturation]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("error opening video file.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = 0
features = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % FRAME_INTERVAL == 0:
        features.append(extract_features(frame))
        total_frames += 1
    frame_idx += 1

cap.release()

feature_cols = ["sharpness", "glare", "contrast", "entropy", "edge_density", "mean_brightness", "saturation"]
df = pd.DataFrame(features, columns=feature_cols)

X_scaled = scaler.transform(df)

y_pred = model.predict(X_scaled)

bad_ratio = np.mean(y_pred == 0)
print(f"Bad frames: {np.sum(y_pred == 0)}/{len(y_pred)} ({bad_ratio:.1%})")

if bad_ratio > REJECTION_THRESHOLD:
    print("Video Rejected.")
else:
    print("Video Accepted.")
