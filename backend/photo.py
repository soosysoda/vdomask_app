import cv2
import easyocr
import re
import numpy as np

# === Configuration ===
IMAGE_PATH = 'WhatsApp Image 2025-05-03 at 13.39.30_e08d7b93.jpg'     # Input image file
OUTPUT_PATH = 'masked_output.jpg'  # Output image file
USE_GPU = False
CONFIDENCE_THRESHOLD = 0.5

# === Initialize OCR Reader ===
reader = easyocr.Reader(['en'], gpu=USE_GPU, model_storage_directory='./models')

# === Regex Patterns ===
aadhaar_patterns = [
    r'\b\d{12}\b',
    r'\b\d{4}\s\d{4}\s\d{4}\b',
    r'\b\d{4}-\d{4}-\d{4}\b'
]
pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'

# === Load Image ===
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not open image file {IMAGE_PATH}")
    exit(1)

frame_height, frame_width = image.shape[:2]

# === OCR Detection ===
results = reader.readtext(image, detail=1, paragraph=False,
                          min_size=10, text_threshold=0.5,
                          link_threshold=0.3, low_text=0.3)

# === Mask Detected Sensitive Info ===
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

# === Save Output ===
cv2.imwrite(OUTPUT_PATH, image)
print(f"Masked image saved to {OUTPUT_PATH}")
