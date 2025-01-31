import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from fpdf import FPDF
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
PDF_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Munsell color mapping
munsell_color_rgb_mapping = {
    "Grayish Pink": (200, 150, 150),
    "Moderate Pink": (210, 140, 140),
    "Pale Red": (220, 130, 130),
    "Light Red": (230, 120, 120),
    "Moderate Red": (180, 80, 80),
    "Grayish Red": (170, 90, 90),
    "Dusky Red": (150, 60, 60),
    "Blackish Red": (140, 50, 50),
    "Very Dark Red": (130, 40, 40),
    "Grayish Orange": (230, 180, 150),
    "Moderate Orange Pink": (240, 170, 140),
    "Pale Yellow": (230, 220, 150),
    "Dark Yellowish Orange": (210, 160, 100),
    "Moderate Brown": (150, 100, 80),
    "Dusky Yellowish Brown": (120, 90, 70),
    "Grayish Yellow": (200, 180, 100),
    "Yellowish Gray": (190, 170, 120),
    "Olive Gray": (150, 140, 110),
    "Dark Greenish Gray": (80, 100, 90),
    "Light Green": (160, 210, 130),
    "Moderate Green": (100, 160, 100),
    "Pale Blue": (180, 200, 230),
    "Grayish Blue": (120, 140, 180),
    "Moderate Blue": (100, 120, 200),
    "Dusky Blue": (90, 110, 160),
    "Very Dusky Purple": (80, 60, 90),
    "Pale Olive": (170, 160, 90),
    "Olive Brown": (120, 110, 80),
    "Dark Olive": (100, 90, 60),
    "Bluish Gray": (100, 120, 140),
    "Light Brownish Gray": (160, 140, 120),
    "Medium Gray": (120, 120, 120),
    "Dark Gray": (80, 80, 80),
    "Black": (30, 30, 30)
}

# Build KDTree
munsell_color_tree = KDTree(list(munsell_color_rgb_mapping.values()))

def closest_munsell_color(dominant_color):
    dominant_color = np.array(dominant_color[::-1])  # Convert BGR to RGB
    _, index = munsell_color_tree.query(dominant_color)
    return list(munsell_color_rgb_mapping.keys())[index]

def classify_cutting(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    shape = "Undefined"
    if aspect_ratio > 3:
        shape = "Very Elongate"
    elif 2 < aspect_ratio <= 3:
        shape = "Elongate"
    elif 1.5 < aspect_ratio <= 2:
        shape = "Sub-Elongate"
    elif 1.2 < aspect_ratio <= 1.5:
        shape = "Sub-Spherical"
    else:
        shape = "Spherical"
    return shape

def detect_cuttings(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 20, 20])
    upper_bound = np.array([180, 255, 180])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    processed_image = image.copy()
    report = []
    
    for i, contour in enumerate(contours, 1):
        if cv2.contourArea(contour) > 300:
            cv2.drawContours(processed_image, [contour], -1, (0, 255, 0), thickness=2)  # Border only
            shape = classify_cutting(contour)
            x, y, w, h = cv2.boundingRect(contour)
            report.append(f"Cutting {i}: Shape={shape}")
    
    processed_image_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
    cv2.imwrite(processed_image_path, processed_image)
    return processed_image_path, report

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            processed_image_path, report = detect_cuttings(filepath)
            return send_file(processed_image_path, as_attachment=True)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
