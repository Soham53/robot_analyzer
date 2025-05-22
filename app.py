from flask import Flask, request
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'received'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.data:
        print("Image received. Processing...")

        nparr = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return "Failed to decode image", 400

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get center pixel brightness (small region around it)
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        region_size = 20  # size of square region to sample around center

        # Crop a small square around the center
        half = region_size // 2
        center_roi = gray[center_y - half:center_y + half, center_x - half:center_x + half]

        # Save the full image and ROI for debugging
        full_img_path = os.path.join(UPLOAD_FOLDER, "full_image.bmp")
        roi_img_path = os.path.join(UPLOAD_FOLDER, "roi_debug.bmp")
        cv2.imwrite(full_img_path, img)
        cv2.imwrite(roi_img_path, center_roi)

        # Calculate median brightness
        median_brightness = np.median(center_roi)
        print("Median brightness at center:", median_brightness)

        if median_brightness > 200:
            return "ON"
        else:
            return "OFF"
    else:
        return "No image uploaded", 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)