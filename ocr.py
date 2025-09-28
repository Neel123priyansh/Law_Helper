import pytesseract
from pdf2image import convert_from_path
import os
import cv2
import numpy as np
from doctr.models import ocr_predictor

#Model

# Path to your scanned PDF
pdf_path = "Test_PDF_For_OCR/D-303-Abode-Valley-Contract-May2025-Signed.pdf"
output_txt = "agreement_text.txt"
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin" 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# Step 1: Convert PDF pages to images
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

#Step 2: Image Preprocessing
def preprocess_image(pil_image):
    # Convert PIL -> OpenCV
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Remove background noise
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Optional: remove small dots
    kernel = np.ones((1,1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img

# Step 3: OCR each page
full_text = ""
for i, page in enumerate(pages):
    clean_pages = preprocess_image(pages)
    text = pytesseract.image_to_string(clean_pages, lang="eng") 
    full_text += f"\n--- Page {i+1} ---\n{text}\n"

# Step 4: Save to .txt file
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ OCR completed. Text saved to {output_txt}")
