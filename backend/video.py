import cv2
import easyocr
import re
import numpy as np
import time
import os

def process_video(video_path, output_path=None):
    # === Configuration ===
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_masked.mp4"

    USE_GPU = False
    OCR_EVERY_N_FRAMES = 10        # Run OCR every N frames
    CONFIDENCE_THRESHOLD = 0.5
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # === Initialize Components ===
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    reader = easyocr.Reader(['en'], gpu=USE_GPU, model_storage_directory='./models')

    aadhaar_patterns = [
        r'\b\d{12}\b',
        r'\b\d{4}\s\d{4}\s\d{4}\b',
        r'\b\d{4}-\d{4}-\d{4}\b'
    ]
    pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'

    frame_count = 0
    trackers = []

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fps = vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    def detect_sensitive_info(frame):
        nonlocal trackers

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = reader.readtext(frame, detail=1, paragraph=False,
                                min_size=10, text_threshold=0.5,
                                link_threshold=0.3, low_text=0.3)

        detected_trackers = []

        for (bbox, text, conf) in results:
            if conf < CONFIDENCE_THRESHOLD:
                continue

            text = text.strip().upper()

            # For Aadhaar: fix OCR confusion of 'O' as '0'
            aadhaar_text = text.replace('O', '0')

            aadhaar_match = any(re.search(pattern, aadhaar_text) for pattern in aadhaar_patterns)
            pan_match = re.search(pan_pattern, text)

            if aadhaar_match or pan_match:
                (tl, tr, br, bl) = bbox
                tl = tuple(map(int, tl))
                br = tuple(map(int, br))
                width = br[0] - tl[0]
                height = br[1] - tl[1]

                padding = 5
                x = max(0, tl[0] - padding)
                y = max(0, tl[1] - padding)
                w = min(FRAME_WIDTH - x, width + padding * 2)
                h = min(FRAME_HEIGHT - y, height + padding * 2)

                rect = (x, y, w, h)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, rect)
                detected_trackers.append(tracker)

        # Replace old trackers with new ones every OCR_EVERY_N_FRAMES
        if detected_trackers:
            trackers.clear()
            trackers.extend(detected_trackers)

    # === Main Loop ===
    start_time = time.time()
    frames_processed = 0

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frames_processed += 1

        display_frame = frame.copy()

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            detect_sensitive_info(frame)

        frame_count += 1

        new_trackers = []
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                x = max(0, x)
                y = max(0, y)
                w = min(FRAME_WIDTH - x, w)
                h = min(FRAME_HEIGHT - y, h)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
                new_trackers.append(tracker)

        trackers = new_trackers
        out.write(display_frame)

    # === Cleanup ===
    total_time = time.time() - start_time
    fps_rate = frames_processed / total_time
    print(f"Processing complete! {frames_processed} frames in {total_time:.2f} seconds ({fps_rate:.2f} FPS)")
    print(f"Output saved to {output_path}")

    vid.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path

# Example usage
if __name__ == "__main__":
    input_video = 'WIN_20250503_12_55_45_Pro (online-video-cutter.com).mp4'
    output_video = process_video(input_video)
