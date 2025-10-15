import cv2
import numpy as np
import matplotlib.pyplot as plt

frame_original = cv2.imread("ACS_frame57.jpg")

def apply_raindrop_overlay(frame, overlay_path, intensity=2.0):
    raindrop = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # Resize
    raindrop = cv2.resize(raindrop, (frame.shape[1], frame.shape[0]))

    # Separate alpha and color channels (this is for transparency)
    alpha = np.clip(raindrop[:, :, 3] / 255.0 * intensity, 0, 1)
    overlay_color = raindrop[:, :, :3]

    # make a copy
    frame_copy = frame.copy()
    for c in range(3):
        frame_copy[:, :, c] = (1 - alpha) * frame_copy[:, :, c] + alpha * overlay_color[:, :, c]

    return cv2.cvtColor(frame_copy.astype(np.uint8), cv2.COLOR_BGR2RGB)

frame_overlay1 = apply_raindrop_overlay(frame_original, "raindrops1.png", intensity=2.0)
frame_overlay2 = apply_raindrop_overlay(frame_original, "raindrops2.png", intensity=1.0)
frame_overlay3 = apply_raindrop_overlay(frame_original, "raindrops3.png", intensity=2.0)
frame_overlay4 = apply_raindrop_overlay(frame_original, "raindrops4.png", intensity=2.0)

# display to see effects
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.imshow(frame_overlay1)
plt.axis('off')
plt.title("Raindrops 1 Overlay")

plt.subplot(2, 2, 2)
plt.imshow(frame_overlay2)
plt.axis('off')
plt.title("Raindrops 2 Overlay")

plt.subplot(2, 2, 3)
plt.imshow(frame_overlay3)
plt.axis('off')
plt.title("Raindrops 3 Overlay")

plt.subplot(2, 2, 4)
plt.imshow(frame_overlay4)
plt.axis('off')
plt.title("Raindrops 4 Overlay")

plt.show()
