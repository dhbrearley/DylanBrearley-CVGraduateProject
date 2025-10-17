import cv2, os

video_dir = "../Videos_Altered"
output_dir = "../Frames_Altered"
os.makedirs(output_dir, exist_ok=True)

for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue
    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    i = 0
    frame_id = 0
    success, frame = cap.read()
    while success:
        if i % 5 == 0:
            fname = f"{os.path.splitext(video_file)[0]}_frame{frame_id}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), frame)
            frame_id += 1
        success, frame = cap.read()
        i += 1
    cap.release()

print("Finished")
