import cv2
import numpy as np

# Read image
img = cv2.imread("melanoma_86.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create black-hat kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))

# Detect dark hairs
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# Threshold to obtain hair mask
_, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

# Remove hairs using inpainting
result = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

cv2.imwrite("hair_removed.png", result)
