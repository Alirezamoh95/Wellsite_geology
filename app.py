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
}

# Build KDTree for fast color lookup
munsell_color_tree = KDTree(list(munsell_color_rgb_mapping.values()))

def closest_munsell_color(dominant_color):
    """Find the closest Munsell color from the predefined mapping"""
    dominant_color = np.array(dominant_color[::-1])  # Convert BGR to RGB
    _, index = munsell_color_tree.query(dominant_color)
    return list(munsell_color_rgb_mapping.keys())[index]

def classify_cutting(contour):
    """Classify shape based on aspect ratio"""
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

def generate_pdf_report(image_path, processed_image_path, report):
    """Generate a PDF report with cutting details"""
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

    pdf_path = os.path.join(PDF_FOLDER, "cuttings_report.pdf")
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

    return pdf_path

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

            # Process image and generate PDF
            pdf_path = detect_cuttings(filepath)

            # Send PDF file as response
            return send_file(pdf_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
