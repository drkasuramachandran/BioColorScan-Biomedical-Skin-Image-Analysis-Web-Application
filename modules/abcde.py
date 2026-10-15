import cv2
import numpy as np


def create_lesion_mask(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def calculate_asymmetry(mask):

    h, w = mask.shape

    left = mask[:, :w // 2]

    right = cv2.flip(
        mask[:, w // 2:],
        1
    )

    minw = min(
        left.shape[1],
        right.shape[1]
    )

    left = left[:, :minw]
    right = right[:, :minw]

    difference = np.sum(
        np.abs(
            left.astype(int) -
            right.astype(int)
        )
    )

    area = np.sum(mask > 0)

    if area == 0:
        return 0.0

    return float(difference / area)


def calculate_circularity(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, None

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    if perimeter == 0:
        return 0.0, cnt

    circularity = (
        4 * np.pi * area /
        (perimeter ** 2)
    )

    return float(circularity), cnt


def calculate_color_variation(
    image,
    mask
):

    pixels = image[mask > 0]

    if len(pixels) == 0:
        return 0.0

    std_rgb = np.std(
        pixels,
        axis=0
    )

    return float(
        np.mean(std_rgb)
    )


def calculate_diameter(
    cnt,
    pixel_size=0.05
):

    if cnt is None:
        return 0.0

    x, y, w, h = cv2.boundingRect(cnt)

    return float(
        max(w, h) * pixel_size
    )


def analyze_abcd(
    image,
    pixel_size=0.05
):

    mask = create_lesion_mask(image)

    asymmetry = calculate_asymmetry(
        mask
    )

    circularity, contour = calculate_circularity(
        mask
    )

    color_variation = calculate_color_variation(
        image,
        mask
    )

    diameter = calculate_diameter(
        contour,
        pixel_size
    )

    score = 0

    if asymmetry > 0.20:
        score += 1

    if circularity < 0.75:
        score += 1

    if color_variation > 30:
        score += 1

    if diameter > 6:
        score += 1

    results = {

        "Asymmetry Index":
            asymmetry,

        "Circularity":
            circularity,

        "Color Variation":
            color_variation,

        "Diameter (mm)":
            diameter,

        "ABCD Score":
            score,

        "Maximum Score":
            4
    }

    return results, mask