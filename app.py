import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify, url_for
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

# Define Munsell color mapping
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

# Build KDTree for fast color lookup
munsell_color_tree = KDTree(list(munsell_color_rgb_mapping.values()))

def closest_munsell_color(dominant_color):
    """Find the closest Munsell color from the predefined mapping"""
    dominant_color = np.array(dominant_color[::-1])  # Convert BGR to RGB
    _, index = munsell_color_tree.query(dominant_color)
    return list(munsell_color_rgb_mapping.keys())[index]



def classify_cutting(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    convex_hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(convex_hull)
    convexity = area / convex_area if convex_area > 0 else 0

    if aspect_ratio > 3:
        shape = "Very Elongate"
    elif 2 < aspect_ratio <= 3:
        shape = "Elongate"
    elif 1.5 < aspect_ratio <= 2:
        shape = "Sub-Elongate"
    elif 1.2 < aspect_ratio <= 1.5:
        shape = "Sub-Spherical"
    elif 0.9 < aspect_ratio <= 1.2:
        shape = "Spherical"
    else:
        shape = "Very Spherical (Equant)"
    
    if circularity > 0.8 and convexity > 0.9:
        roundness = "Well Rounded"
    elif circularity > 0.7:
        roundness = "Rounded"
    elif circularity > 0.6:
        roundness = "Subrounded"
    elif circularity > 0.5:
        roundness = "Subangular"
    elif circularity > 0.4:
        roundness = "Angular"
    else:
        roundness = "Very Angular"

    return shape,roundness


def generate_pdf_report(image_path, processed_image_path, report):
    """Generate PDF report with cuttings details."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Cuttings Detection Report", ln=True, align='C')

    pdf.image(image_path, x=10, y=30, w=180)
    pdf.ln(100)
    pdf.image(processed_image_path, x=10, y=140, w=180)
    pdf.ln(100)

    pdf.add_page()
    pdf.cell(200, 10, "Cuttings Description", ln=True, align='C')
    pdf.ln(10)
    for desc in report:
        pdf.multi_cell(0, 10, desc)

    pdf_path = os.path.join(PROCESSED_FOLDER, "cuttings_report.pdf")  # Save in "static/processed/"

    pdf.output(pdf_path)
    return pdf_path

def detect_cuttings(image_path):
    """Detect cuttings in the image and analyze their properties"""
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

            # Extract dominant color
            x, y, w, h = cv2.boundingRect(contour)
            cutting_mask = np.zeros_like(mask)
            cv2.drawContours(cutting_mask, [contour], -1, 255, thickness=cv2.FILLED)
            masked_pixels = image[cutting_mask > 0]
            kmeans = KMeans(n_clusters=1, n_init=10).fit(masked_pixels)
            dominant_color = tuple(map(int, kmeans.cluster_centers_[0]))
            munsell_color = closest_munsell_color(dominant_color)

            # Annotate image
            cv2.putText(processed_image, str(i), (x + w//2, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            report.append(f"Cutting {i}: Shape={shape}, Color={munsell_color}")

    processed_image_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
    cv2.imwrite(processed_image_path, processed_image)

    # Generate PDF Report
    pdf_path = generate_pdf_report(image_path, processed_image_path, report)

    return processed_image_path,pdf_path

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

            processed_image_path, pdf_path = detect_cuttings(filepath)
            """
            return jsonify({
                "processed_image_url": url_for('static', filename='processed/processed_image.png'),
                "pdf_url": url_for('download_pdf')  # Correctly serve the PDF
            })
            """
            return jsonify({
                "processed_image_url": request.host_url + "static/processed/processed_image.png",
                "pdf_url": request.host_url + "download-pdf"
            })

    return render_template("index.html")

@app.route("/download-pdf")
def download_pdf():
    pdf_path = os.path.join(PROCESSED_FOLDER, "cuttings_report.pdf")
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
