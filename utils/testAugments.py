import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# === CONFIG ===
IMAGE_PATH = "ACS_frame57.jpg"  # Change to your image
MIN_DARKEN, MAX_DARKEN = -70, -120
MIN_BRIGHTEN, MAX_BRIGHTEN = 90, 120
MIN_BLUR, MAX_BLUR = 37, 129  # Odd numbers only

# === FUNCTIONS ===

def adjust_brightness(frame, value):
    """Adjust brightness using HSV, safely handling types and clipping."""
    if frame is None:
        raise ValueError("Input frame is None (check image path).")
    
    # Convert to HSV (ensure uint8)
    hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply brightness offset
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    
    # Merge back and convert
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_random_brightness(frame):
    if random.random() < 0.5:
        value = random.randint(MIN_DARKEN, MAX_DARKEN)
        reason = f"Darkened ({value})"
    else:
        value = random.randint(MIN_BRIGHTEN, MAX_BRIGHTEN)
        reason = f"Brightened (+{value})"
    return adjust_brightness(frame, value), reason

def apply_random_blur(frame):
    k = random.choice(range(MIN_BLUR, MAX_BLUR + 1, 2))
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return blurred, f"Blurred (kernel={k})"

# === LOAD IMAGE ===
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# === GENERATE TEST VARIANTS ===
tests = []

# Test multiple brightness adjustments
for val in [MIN_DARKEN, -25, MAX_DARKEN, MIN_BRIGHTEN, 25, MAX_BRIGHTEN]:
    adj = adjust_brightness(frame, val)
    tests.append((adj, f"Brightness {val}"))

# Test multiple blur levels
for k in [3, 5, 9, MIN_BLUR, MAX_BLUR]:
    k = k if k % 2 == 1 else k + 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    tests.append((blurred, f"Blur kernel={k}"))

# === DISPLAY RESULTS ===
plt.figure(figsize=(15, 10))
for i, (img, title) in enumerate(tests):
    plt.subplot(3, 4, i + 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(title)

plt.tight_layout()
plt.show()
