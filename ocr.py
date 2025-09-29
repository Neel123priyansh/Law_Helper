import pytesseract
from pdf2image import convert_from_path
import os
import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Model
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

pdf_path = "Test_PDF_For_OCR/D-303-Abode-Valley-Contract-May2025-Signed.pdf"
output_txt = "agreement_text.txt"

# # Step 1: Convert PDF pages to images
# pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
# pdf_doc = DocumentFile.from_pdf(pdf_path)

# Step 1: Load PDF into docTR
pdf_doc = DocumentFile.from_pdf(pdf_path)

# Step 2: Run OCR
result = model(pdf_doc)

# Step 3: Export to plain text
full_text = result.render()

# Step 4: Save to .txt file
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(full_text)

print("✅ OCR completed with docTR. Text saved to", output_txt)
