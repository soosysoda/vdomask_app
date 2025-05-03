from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import easyocr
import re
import numpy as np
import time
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Shared Configuration ===
USE_GPU = True
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Initialize OCR Reader
reader = easyocr.Reader(['en'], gpu=USE_GPU, model_storage_directory='./models')

# Regex Patterns
aadhaar_patterns = [
    r'\b\d{12}\b',
    r'\b\d{4}\s\d{4}\s\d{4}\b',
    r'\b\d{4}-\d{4}-\d{4}\b'
]
pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        return None

    frame_height, frame_width = image.shape[:2]

    # OCR Detection
    results = reader.readtext(image, detail=1, paragraph=False,
                            min_size=10, text_threshold=0.5,
                            link_threshold=0.3, low_text=0.3)

    # Mask Detected Sensitive Info
    for (bbox, text, conf) in results:
        if conf < CONFIDENCE_THRESHOLD:
            continue

        text = text.strip().upper()
        aadhaar_text = text.replace('O', '0')  # Handle 'O' vs '0'

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
            w = min(frame_width - x, width + padding * 2)
            h = min(frame_height - y, height + padding * 2)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

    output_path = os.path.join(OUTPUT_FOLDER, 'masked_' + os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return output_path

def process_video(video_path):
    # Initialize Components
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        return None

    output_path = os.path.join(OUTPUT_FOLDER, 'masked_' + os.path.basename(video_path))
    
    frame_count = 0
    trackers = []

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fps = vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    def detect_sensitive_info(frame):
        nonlocal trackers

        results = reader.readtext(frame, detail=1, paragraph=False,
                                min_size=10, text_threshold=0.5,
                                link_threshold=0.3, low_text=0.3)

        detected_trackers = []

        for (bbox, text, conf) in results:
            if conf < CONFIDENCE_THRESHOLD:
                continue

            text = text.strip().upper()
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

        if detected_trackers:
            trackers.clear()
            trackers.extend(detected_trackers)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display_frame = frame.copy()

        if frame_count % 10 == 0:  # Process every 10th frame
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

    vid.release()
    out.release()
    return output_path

logging.basicConfig(level=logging.INFO)

@app.route('/process/image', methods=['POST'])
def process_image_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'File must be an image (jpg, jpeg, or png)'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    output_path = process_image(file_path)
    if output_path is None:
        logging.error(f"Image processing failed for {file_path}")
        return jsonify({'error': 'Image processing failed'}), 500
    return send_file(output_path, as_attachment=True)

@app.route('/process/video', methods=['POST'])
def process_video_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.mp4'):
        return jsonify({'error': 'File must be an MP4 video'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    output_path = process_video(file_path)
    if output_path is None:
        return jsonify({'error': 'Video processing failed'}), 500

    return send_file(output_path, as_attachment=True, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True, port=5000)