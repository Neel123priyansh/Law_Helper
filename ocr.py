import cv2
import pytesseract
import numpy as np

# Path to tesseract (for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocrcore(img):
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    return text

# --- Preprocessing Functions ---
def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 3) 

def thresholding(image):

    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2)

def deskew(image):
    coords = cv2.findNonZero(cv2.bitwise_not(image))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def upscale(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def morph_open_close(image):
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

# --- Main Execution ---
img = cv2.imread("ocr_ss.png")

img = get_greyscale(img)
img = upscale(img)
img = thresholding(img)
img = remove_noise(img)
img = deskew(img)
img = morph_open_close(img)


text = ocrcore(img)
print(text)
