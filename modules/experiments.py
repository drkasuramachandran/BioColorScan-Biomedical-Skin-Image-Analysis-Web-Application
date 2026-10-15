# ============================================================
# SKIN LESION RESEARCH PLATFORM
# EXPERIMENTS A - E
# ============================================================
#
# Experiment A:
#   ImageJ ROI
#       ↓
#   ROI-only Shades of Gray (p=6)
#       ↓
#   CIELAB
#       ↓
#   Mean L*, a*, b*
#       ↓
#   ITA
#
# Experiment B:
#   ImageJ ROI
#       ↓
#   ROI bounding-box crop
#       ↓
#   Shades of Gray (p=6) using entire crop
#       ↓
#   CIELAB
#       ↓
#   Mean L*, a*, b*
#       ↓
#   ITA
#
# Experiment C:
#   ImageJ ROI
#       ↓
#   ROI-only Shades of Gray (p=6)
#       ↓
#   Method 1: CIELAB → ITA
#
#   Method 2:
#       CIELAB → L* stretching → CIELAB → ITA
#
# Experiment D:
#   ImageJ ROI
#       ↓
#   Hair removal
#       ↓
#   CIELAB
#       ↓
#   Mean L*, a*, b*
#       ↓
#   ITA
#
# Experiment E:
#   ImageJ ROI
#       ↓
#   ROI-only Shades of Gray (p=6)
#       ↓
#   CIELAB → ITA
#       ↓
#   Hair removal
#       ↓
#   CIELAB → ITA
#
# IMPORTANT:
#   ROI1, ROI2 and ROI3 are retained independently.
#   No ROI pixel subtraction is performed in Experiments A-E.
#
# ============================================================


import io
import zipfile

import cv2
import numpy as np

from roifile import ImagejRoi


# ============================================================
# GENERAL SETTINGS
# ============================================================

SOG_P = 6

# Hair-removal settings
KERNEL_SIZES = [9, 13]
ANGLES = range(0, 180, 15)

ASPECT_RATIO_THRESHOLD = 1.4
MAX_AREA_FRACTION = 0.03
MIN_AREA = 4

ADAPTIVE_BLOCK_SIZE = 15
ADAPTIVE_C = -2

INPAINT_RADIUS = 3


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image):

    if image is None:
        raise ValueError("Input image is empty.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Input image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Input image contains no pixels."
        )

    if image.dtype != np.uint8:
        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    return image


# ============================================================
# IMAGEJ ROI TYPE
# ============================================================

def get_roi_type(roi):

    try:
        return str(roi.roitype)
    except Exception:
        return "Unknown"


# ============================================================
# IMAGEJ ROI TO MASK
# ============================================================

def roi_to_mask(roi, image_shape):

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    roi_type = get_roi_type(roi).lower()

    # --------------------------------------------------------
    # RECTANGLE
    # --------------------------------------------------------

    if "rectangle" in roi_type:

        try:

            rectangle = roi.rectangle

            x0 = int(round(rectangle.left))
            y0 = int(round(rectangle.top))
            x1 = int(round(rectangle.right))
            y1 = int(round(rectangle.bottom))

            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))

            if x1 >= x0 and y1 >= y0:

                cv2.rectangle(
                    mask,
                    (x0, y0),
                    (x1, y1),
                    255,
                    thickness=-1
                )

                return mask

        except Exception:
            pass

    # --------------------------------------------------------
    # OVAL
    # --------------------------------------------------------

    if "oval" in roi_type:

        try:

            rectangle = roi.rectangle

            x0 = int(round(rectangle.left))
            y0 = int(round(rectangle.top))
            x1 = int(round(rectangle.right))
            y1 = int(round(rectangle.bottom))

            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))

            center_x = int(round((x0 + x1) / 2))
            center_y = int(round((y0 + y1) / 2))

            radius_x = max(
                1,
                int(round((x1 - x0) / 2))
            )

            radius_y = max(
                1,
                int(round((y1 - y0) / 2))
            )

            cv2.ellipse(
                mask,
                (center_x, center_y),
                (radius_x, radius_y),
                0,
                0,
                360,
                255,
                thickness=-1
            )

            return mask

        except Exception:
            pass

    # --------------------------------------------------------
    # POLYGON / FREEHAND / COORDINATE ROI
    # --------------------------------------------------------

    try:

        coordinates = roi.coordinates()

        if coordinates is not None:

            coordinates = np.asarray(
                coordinates
            )

            if (
                coordinates.ndim == 2
                and coordinates.shape[1] >= 2
            ):

                points = coordinates[:, :2]

                points = np.round(
                    points
                ).astype(np.int32)

                points[:, 0] = np.clip(
                    points[:, 0],
                    0,
                    width - 1
                )

                points[:, 1] = np.clip(
                    points[:, 1],
                    0,
                    height - 1
                )

                if len(points) >= 3:

                    cv2.fillPoly(
                        mask,
                        [
                            points.reshape(
                                (-1, 1, 2)
                            )
                        ],
                        255
                    )

                    return mask

    except Exception:
        pass

    return mask


# ============================================================
# LOAD IMAGEJ ROIS FROM ZIP
# ============================================================

def load_imagej_rois(roi_zip):

    if roi_zip is None:
        raise ValueError(
            "ROI ZIP file is empty."
        )

    # --------------------------------------------------------
    # Get ZIP bytes
    # --------------------------------------------------------

    if isinstance(roi_zip, bytes):

        zip_data = roi_zip

    elif hasattr(roi_zip, "getvalue"):

        zip_data = roi_zip.getvalue()

    elif hasattr(roi_zip, "read"):

        zip_data = roi_zip.read()

    else:

        raise TypeError(
            "ROI ZIP must be bytes or a file-like object."
        )

    # --------------------------------------------------------
    # Read ROI files
    # --------------------------------------------------------

    rois = []

    with zipfile.ZipFile(
        io.BytesIO(zip_data),
        "r"
    ) as zip_file:

        roi_files = [
            name
            for name in zip_file.namelist()
            if name.lower().endswith(".roi")
        ]

        roi_files.sort(
            key=lambda name: name.lower()
        )

        if len(roi_files) == 0:

            raise ValueError(
                "No .roi files were found in the ZIP file."
            )

        for filename in roi_files:

            roi_bytes = zip_file.read(
                filename
            )

            roi = ImagejRoi.frombytes(
                roi_bytes
            )

            rois.append(
                (filename, roi)
            )

    return rois


# ============================================================
# CROP IMAGE TO ROI BOUNDING BOX
# ============================================================

def crop_to_roi(image, mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        raise ValueError(
            "ROI contains no pixels."
        )

    x_min = np.min(xs)
    x_max = np.max(xs)

    y_min = np.min(ys)
    y_max = np.max(ys)

    cropped_image = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    cropped_mask = mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return (
        cropped_image,
        cropped_mask
    )


# ============================================================
# CIELAB CONVERSION
# ============================================================

def convert_to_cielab(image):

    image = validate_image(
        image
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(
        np.float32
    )

    L = (
        lab[:, :, 0]
        *
        (100.0 / 255.0)
    )

    a = (
        lab[:, :, 1]
        -
        128.0
    )

    b = (
        lab[:, :, 2]
        -
        128.0
    )

    return L, a, b


# ============================================================
# ITA FROM MEAN LAB
# ============================================================

def calculate_ita_from_mean_lab(
    mean_L,
    mean_b
):

    if abs(mean_b) < 1e-12:

        if mean_L > 50:
            return 90.0

        elif mean_L < 50:
            return -90.0

        else:
            return 0.0

    return float(
        np.degrees(
            np.arctan(
                (mean_L - 50.0)
                /
                mean_b
            )
        )
    )


# ============================================================
# CALCULATE MEAN LAB + ITA
# ============================================================

def calculate_mean_lab_ita(
    image,
    mask
):

    L, a, b = convert_to_cielab(
        image
    )

    mask_bool = (
        mask > 0
    )

    if not np.any(mask_bool):

        raise ValueError(
            "ROI contains no pixels."
        )

    L_roi = L[
        mask_bool
    ]

    a_roi = a[
        mask_bool
    ]

    b_roi = b[
        mask_bool
    ]

    valid = (
        np.isfinite(L_roi)
        &
        np.isfinite(a_roi)
        &
        np.isfinite(b_roi)
    )

    L_roi = L_roi[
        valid
    ]

    a_roi = a_roi[
        valid
    ]

    b_roi = b_roi[
        valid
    ]

    if len(L_roi) == 0:

        raise ValueError(
            "No valid ROI pixels were found."
        )

    mean_L = float(
        np.mean(L_roi)
    )

    mean_a = float(
        np.mean(a_roi)
    )

    mean_b = float(
        np.mean(b_roi)
    )

    ita = calculate_ita_from_mean_lab(
        mean_L,
        mean_b
    )

    return {
        "Pixel Count": int(len(L_roi)),
        "Mean L*": mean_L,
        "Mean a*": mean_a,
        "Mean b*": mean_b,
        "ITA": ita
    }


# ============================================================
# PIXEL-WISE ITA STATISTICS
# ============================================================

def calculate_pixelwise_ita_statistics(
    image,
    mask
):

    L, a, b = convert_to_cielab(
        image
    )

    mask_bool = (
        mask > 0
    )

    if not np.any(mask_bool):

        return {
            "Pixel-wise Min ITA": np.nan,
            "Pixel-wise Max ITA": np.nan,
            "Pixel-wise Mean ITA": np.nan,
            "Pixel-wise Median ITA": np.nan,
            "Pixel-wise Std. Dev.": np.nan
        }

    L_roi = L[
        mask_bool
    ]

    b_roi = b[
        mask_bool
    ]

    valid = (
        np.isfinite(L_roi)
        &
        np.isfinite(b_roi)
        &
        (np.abs(b_roi) > 1e-12)
    )

    L_roi = L_roi[
        valid
    ]

    b_roi = b_roi[
        valid
    ]

    if len(L_roi) == 0:

        return {
            "Pixel-wise Min ITA": np.nan,
            "Pixel-wise Max ITA": np.nan,
            "Pixel-wise Mean ITA": np.nan,
            "Pixel-wise Median ITA": np.nan,
            "Pixel-wise Std. Dev.": np.nan
        }

    ita_values = np.degrees(
        np.arctan2(
            L_roi - 50.0,
            b_roi
        )
    )

    return {
        "Pixel-wise Min ITA":
            float(np.min(ita_values)),

        "Pixel-wise Max ITA":
            float(np.max(ita_values)),

        "Pixel-wise Mean ITA":
            float(np.mean(ita_values)),

        "Pixel-wise Median ITA":
            float(np.median(ita_values)),

        "Pixel-wise Std. Dev.":
            float(np.std(ita_values))
    }


# ============================================================
# SHADES OF GRAY
# ============================================================

def shades_of_gray(
    image,
    p=6
):

    image = validate_image(
        image
    )

    image_float = image.astype(
        np.float32
    )

    # Avoid zero values causing problems
    image_float = np.maximum(
        image_float,
        1.0
    )

    illumination = (
        np.mean(
            np.abs(image_float) ** p,
            axis=(0, 1)
        )
        ** (1.0 / p)
    )

    illumination_mean = np.mean(
        illumination
    )

    if illumination_mean <= 0:

        raise ValueError(
            "Unable to estimate illumination."
        )

    illumination = (
        illumination
        /
        illumination_mean
    )

    corrected = (
        image_float
        /
        illumination
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(
        np.uint8
    )

    return corrected


# ============================================================
# ROI-ONLY SHADES OF GRAY
# ============================================================

def shades_of_gray_roi(
    image,
    mask,
    p=6
):

    image = validate_image(
        image
    )

    mask_bool = (
        mask > 0
    )

    if not np.any(mask_bool):

        raise ValueError(
            "ROI contains no pixels."
        )

    image_float = image.astype(
        np.float32
    )

    roi_pixels = image_float[
        mask_bool
    ]

    roi_pixels = np.maximum(
        roi_pixels,
        1.0
    )

    illumination = (
        np.mean(
            np.abs(roi_pixels) ** p,
            axis=0
        )
        ** (1.0 / p)
    )

    illumination_mean = np.mean(
        illumination
    )

    if illumination_mean <= 0:

        raise ValueError(
            "Unable to estimate ROI illumination."
        )

    illumination = (
        illumination
        /
        illumination_mean
    )

    corrected = (
        image_float
        /
        illumination
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(
        np.uint8
    )

    return corrected


# ============================================================
# EXPERIMENT C - L* STRETCHING
# ============================================================

def L_stretch(
    image,
    mask
):

    image = validate_image(
        image
    )

    mask_bool = (
        mask > 0
    )

    if not np.any(mask_bool):

        raise ValueError(
            "ROI contains no pixels."
        )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    L = lab[
        :, :, 0
    ].astype(
        np.float32
    )

    roi_L = L[
        mask_bool
    ]

    L_min = np.min(
        roi_L
    )

    L_max = np.max(
        roi_L
    )

    if L_max > L_min:

        L_stretched = (
            (
                L - L_min
            )
            /
            (
                L_max - L_min
            )
        ) * 255.0

    else:

        L_stretched = L.copy()

    L_stretched = np.clip(
        L_stretched,
        0,
        255
    ).astype(
        np.uint8
    )

    lab[:, :, 0] = (
        L_stretched
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# HAIR REMOVAL
# ============================================================

def detect_hair_mask(
    image
):
    """
    Detect hair-like structures using black-hat morphology,
    directional line kernels and adaptive thresholding.
    """

    image = validate_image(
        image
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    combined_mask = np.zeros_like(
        gray
    )

    height, width = gray.shape

    image_area = (
        height * width
    )

    max_area = (
        image_area
        *
        MAX_AREA_FRACTION
    )

    # --------------------------------------------------------
    # Directional black-hat detection
    # --------------------------------------------------------

    for kernel_size in KERNEL_SIZES:

        for angle in ANGLES:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (
                    kernel_size,
                    1
                )
            )

            center = (
                kernel_size // 2,
                0
            )

            rotation_matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            rotated_kernel = cv2.warpAffine(
                kernel,
                rotation_matrix,
                (
                    kernel_size,
                    kernel_size
                ),
                flags=cv2.INTER_NEAREST
            )

            blackhat = cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                rotated_kernel
            )

            threshold_value = np.percentile(
                blackhat,
                97
            )

            if threshold_value <= 0:
                continue

            _, local_mask = cv2.threshold(
                blackhat,
                threshold_value,
                255,
                cv2.THRESH_BINARY
            )

            combined_mask = cv2.bitwise_or(
                combined_mask,
                local_mask
            )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    combined_mask = cv2.bitwise_and(
        combined_mask,
        adaptive_mask
    )

    # --------------------------------------------------------
    # Contour filtering
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_mask = np.zeros_like(
        gray
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        if area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if h == 0 or w == 0:
            continue

        aspect_ratio = (
            max(w, h)
            /
            max(
                1,
                min(w, h)
            )
        )

        if aspect_ratio < ASPECT_RATIO_THRESHOLD:
            continue

        cv2.drawContours(
            filtered_mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

    # --------------------------------------------------------
    # Small dilation
    # --------------------------------------------------------

    dilation_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    filtered_mask = cv2.dilate(
        filtered_mask,
        dilation_kernel,
        iterations=1
    )

    return filtered_mask


# ============================================================
# HAIR REMOVAL + INPAINTING
# ============================================================

def remove_hair(
    image
):

    image = validate_image(
        image
    )

    hair_mask = detect_hair_mask(
        image
    )

    if not np.any(
        hair_mask > 0
    ):

        return (
            image.copy(),
            hair_mask
        )

    inpainted = cv2.inpaint(
        image,
        hair_mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA
    )

    return (
        inpainted,
        hair_mask
    )


# ============================================================
# APPLY HAIR REMOVAL ONLY INSIDE ROI
# ============================================================

def remove_hair_in_roi(
    image,
    roi_mask
):

    image = validate_image(
        image
    )

    roi_bool = (
        roi_mask > 0
    )

    if not np.any(roi_bool):

        raise ValueError(
            "ROI contains no pixels."
        )

    # --------------------------------------------------------
    # Crop to ROI bounding box
    # --------------------------------------------------------

    cropped_image, cropped_mask = (
        crop_to_roi(
            image,
            roi_mask
        )
    )

    # --------------------------------------------------------
    # Detect hair in cropped ROI region
    # --------------------------------------------------------

    hair_mask = detect_hair_mask(
        cropped_image
    )

    # --------------------------------------------------------
    # Restrict hair mask to ROI
    # --------------------------------------------------------

    hair_mask = cv2.bitwise_and(
        hair_mask,
        cropped_mask
    )

    # --------------------------------------------------------
    # Inpaint
    # --------------------------------------------------------

    if np.any(
        hair_mask > 0
    ):

        corrected_crop = cv2.inpaint(
            cropped_image,
            hair_mask,
            INPAINT_RADIUS,
            cv2.INPAINT_TELEA
        )

    else:

        corrected_crop = (
            cropped_image.copy()
        )

    return (
        corrected_crop,
        cropped_mask,
        hair_mask
    )


# ============================================================
# EXPERIMENT A
# ============================================================

def analyze_experiment_a(
    image_bgr,
    roi_zip,
    p=6
):
    """
    Experiment A:

    ImageJ ROI
        ↓
    ROI-only Shades of Gray
        ↓
    CIELAB
        ↓
    Mean Lab
        ↓
    ITA

    Pixel-wise ITA statistics are also returned.
    """

    image_bgr = validate_image(
        image_bgr
    )

    roi_list = load_imagej_rois(
        roi_zip
    )

    results = []
    processed_rois = []

    for index, (
        roi_filename,
        roi
    ) in enumerate(roi_list):

        full_mask = roi_to_mask(
            roi,
            image_bgr.shape
        )

        cropped_image, cropped_mask = (
            crop_to_roi(
                image_bgr,
                full_mask
            )
        )

        corrected_image = (
            shades_of_gray_roi(
                cropped_image,
                cropped_mask,
                p=p
            )
        )

        mean_results = (
            calculate_mean_lab_ita(
                corrected_image,
                cropped_mask
            )
        )

        pixel_results = (
            calculate_pixelwise_ita_statistics(
                corrected_image,
                cropped_mask
            )
        )

        if index == 0:
            region_name = (
                "ROI 1 — Lesion"
            )
        elif index == 1:
            region_name = (
                "ROI 2 — Closed Lesion Region"
            )
        elif index == 2:
            region_name = (
                "ROI 3 — Healthy Region"
            )
        else:
            region_name = (
                f"ROI {index + 1}"
            )

        result = {
            "Region": region_name,
            "Original ROI File": roi_filename,
            "ROI Type": get_roi_type(roi),
            "Pixel Count": mean_results["Pixel Count"],
            "Mean L*": mean_results["Mean L*"],
            "Mean a*": mean_results["Mean a*"],
            "Mean b*": mean_results["Mean b*"],
            "ITA": mean_results["ITA"],
            "Pixel-wise Min ITA":
                pixel_results["Pixel-wise Min ITA"],
            "Pixel-wise Max ITA":
                pixel_results["Pixel-wise Max ITA"],
            "Pixel-wise Mean ITA":
                pixel_results["Pixel-wise Mean ITA"],
            "Pixel-wise Median ITA":
                pixel_results["Pixel-wise Median ITA"],
            "Pixel-wise Std. Dev.":
                pixel_results["Pixel-wise Std. Dev."]
        }

        results.append(
            result
        )

        processed_rois.append({
            "region": region_name,
            "roi_file": roi_filename,
            "mask": cropped_mask,
            "corrected_image": corrected_image
        })

    return {
        "experiment": "A",
        "p": p,
        "results": results,
        "processed_rois": processed_rois
    }


# ============================================================
# EXPERIMENT B
# ============================================================

def analyze_experiment_b(
    image_bgr,
    roi_zip,
    p=6
):
    """
    Experiment B:

    ImageJ ROI
        ↓
    ROI bounding-box crop
        ↓
    Shades of Gray using the entire crop
        ↓
    CIELAB
        ↓
    Mean Lab
        ↓
    ITA

    No ROI subtraction.
    """

    image_bgr = validate_image(
        image_bgr
    )

    roi_list = load_imagej_rois(
        roi_zip
    )

    results = []
    processed_rois = []

    for index, (
        roi_filename,
        roi
    ) in enumerate(roi_list):

        full_mask = roi_to_mask(
            roi,
            image_bgr.shape
        )

        cropped_image, cropped_mask = (
            crop_to_roi(
                image_bgr,
                full_mask
            )
        )

        # ----------------------------------------------------
        # Experiment B:
        # SoG uses the ENTIRE bounding-box crop
        # ----------------------------------------------------

        corrected_image = (
            shades_of_gray(
                cropped_image,
                p=p
            )
        )

        mean_results = (
            calculate_mean_lab_ita(
                corrected_image,
                cropped_mask
            )
        )

        if index == 0:
            region_name = (
                "ROI 1 — Lesion"
            )
        elif index == 1:
            region_name = (
                "ROI 2 — Closed Lesion Region"
            )
        elif index == 2:
            region_name = (
                "ROI 3 — Healthy Region"
            )
        else:
            region_name = (
                f"ROI {index + 1}"
            )

        result = {
            "Region": region_name,
            "Original ROI File": roi_filename,
            "ROI Type": get_roi_type(roi),
            "Pixel Count": mean_results["Pixel Count"],
            "Mean L*": mean_results["Mean L*"],
            "Mean a*": mean_results["Mean a*"],
            "Mean b*": mean_results["Mean b*"],
            "ITA": mean_results["ITA"]
        }

        results.append(
            result
        )

        processed_rois.append({
            "region": region_name,
            "roi_file": roi_filename,
            "mask": cropped_mask,
            "corrected_image": corrected_image
        })

    return {
        "experiment": "B",
        "p": p,
        "results": results,
        "processed_rois": processed_rois
    }


# ============================================================
# EXPERIMENT C
# ============================================================

def analyze_experiment_c(
    image_bgr,
    roi_zip,
    p=6
):
    """
    Experiment C:

    Method 1:
        ROI-only SoG
        ↓
        CIELAB
        ↓
        ITA

    Method 2:
        ROI-only SoG
        ↓
        L* stretching
        ↓
        CIELAB
        ↓
        ITA
    """

    image_bgr = validate_image(
        image_bgr
    )

    roi_list = load_imagej_rois(
        roi_zip
    )

    results = []
    processed_rois = []

    for index, (
        roi_filename,
        roi
    ) in enumerate(roi_list):

        full_mask = roi_to_mask(
            roi,
            image_bgr.shape
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropped_image, cropped_mask = (
            crop_to_roi(
                image_bgr,
                full_mask
            )
        )

        # ----------------------------------------------------
        # ROI-only Shades of Gray
        # ----------------------------------------------------

        sog_image = (
            shades_of_gray_roi(
                cropped_image,
                cropped_mask,
                p=p
            )
        )

        # ----------------------------------------------------
        # METHOD 1
        #
        # SoG → CIELAB → ITA
        # ----------------------------------------------------

        method1_results = (
            calculate_mean_lab_ita(
                sog_image,
                cropped_mask
            )
        )

        # ----------------------------------------------------
        # METHOD 2
        #
        # SoG → L stretching → CIELAB → ITA
        # ----------------------------------------------------

        stretched_image = (
            L_stretch(
                sog_image,
                cropped_mask
            )
        )

        method2_results = (
            calculate_mean_lab_ita(
                stretched_image,
                cropped_mask
            )
        )

        if index == 0:
            region_name = (
                "ROI 1 — Lesion"
            )
        elif index == 1:
            region_name = (
                "ROI 2 — Closed Lesion Region"
            )
        elif index == 2:
            region_name = (
                "ROI 3 — Healthy Region"
            )
        else:
            region_name = (
                f"ROI {index + 1}"
            )

        result = {
            "Region": region_name,
            "Original ROI File": roi_filename,
            "ROI Type": get_roi_type(roi),
            "Pixel Count":
                method1_results["Pixel Count"],

            # ------------------------------------------------
            # Method 1
            # ------------------------------------------------

            "Method 1 Mean L*":
                method1_results["Mean L*"],

            "Method 1 Mean a*":
                method1_results["Mean a*"],

            "Method 1 Mean b*":
                method1_results["Mean b*"],

            "Method 1 ITA":
                method1_results["ITA"],

            # ------------------------------------------------
            # Method 2
            # ------------------------------------------------

            "Method 2 Mean L*":
                method2_results["Mean L*"],

            "Method 2 Mean a*":
                method2_results["Mean a*"],

            "Method 2 Mean b*":
                method2_results["Mean b*"],

            "Method 2 ITA":
                method2_results["ITA"]
        }

        results.append(
            result
        )

        processed_rois.append({
            "region": region_name,
            "roi_file": roi_filename,
            "mask": cropped_mask,
            "sog_image": sog_image,
            "stretched_image":
                stretched_image
        })

    return {
        "experiment": "C",
        "p": p,
        "results": results,
        "processed_rois": processed_rois
    }


# ============================================================
# EXPERIMENT D
# ============================================================

def analyze_experiment_d(
    image_bgr,
    roi_zip
):
    """
    Experiment D:

    ImageJ ROI
        ↓
    Hair removal
        ↓
    CIELAB
        ↓
    Mean Lab
        ↓
    ITA

    Hair removal is performed independently
    for each ROI.
    """

    image_bgr = validate_image(
        image_bgr
    )

    roi_list = load_imagej_rois(
        roi_zip
    )

    results = []
    processed_rois = []

    for index, (
        roi_filename,
        roi
    ) in enumerate(roi_list):

        full_mask = roi_to_mask(
            roi,
            image_bgr.shape
        )

        # ----------------------------------------------------
        # Hair removal only inside current ROI
        # ----------------------------------------------------

        hair_removed_image, cropped_mask, hair_mask = (
            remove_hair_in_roi(
                image_bgr,
                full_mask
            )
        )

        # ----------------------------------------------------
        # CIELAB + ITA
        # ----------------------------------------------------

        mean_results = (
            calculate_mean_lab_ita(
                hair_removed_image,
                cropped_mask
            )
        )

        if index == 0:
            region_name = (
                "ROI 1 — Lesion"
            )
        elif index == 1:
            region_name = (
                "ROI 2 — Closed Lesion Region"
            )
        elif index == 2:
            region_name = (
                "ROI 3 — Healthy Region"
            )
        else:
            region_name = (
                f"ROI {index + 1}"
            )

        result = {
            "Region": region_name,
            "Original ROI File": roi_filename,
            "ROI Type": get_roi_type(roi),
            "Pixel Count":
                mean_results["Pixel Count"],
            "Mean L*":
                mean_results["Mean L*"],
            "Mean a*":
                mean_results["Mean a*"],
            "Mean b*":
                mean_results["Mean b*"],
            "ITA":
                mean_results["ITA"]
        }

        results.append(
            result
        )

        processed_rois.append({
            "region": region_name,
            "roi_file": roi_filename,
            "mask": cropped_mask,
            "hair_mask": hair_mask,
            "hair_removed_image":
                hair_removed_image
        })

    return {
        "experiment": "D",
        "results": results,
        "processed_rois": processed_rois
    }


# ============================================================
# EXPERIMENT E
# ============================================================

def analyze_experiment_e(
    image_bgr,
    roi_zip,
    p=6
):
    """
    Experiment E:

    Original
        ↓
    ROI-only illumination correction
        ↓
    CIELAB
        ↓
    ITA
        ↓
    Hair removal
        ↓
    CIELAB
        ↓
    ITA

    The two ITA measurements are kept separate.
    """

    image_bgr = validate_image(
        image_bgr
    )

    roi_list = load_imagej_rois(
        roi_zip
    )

    results = []
    processed_rois = []

    for index, (
        roi_filename,
        roi
    ) in enumerate(roi_list):

        full_mask = roi_to_mask(
            roi,
            image_bgr.shape
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropped_image, cropped_mask = (
            crop_to_roi(
                image_bgr,
                full_mask
            )
        )

        # ----------------------------------------------------
        # STEP 1
        #
        # ROI-only illumination correction
        # ----------------------------------------------------

        illumination_corrected = (
            shades_of_gray_roi(
                cropped_image,
                cropped_mask,
                p=p
            )
        )

        # ----------------------------------------------------
        # ITA after illumination correction
        # ----------------------------------------------------

        illumination_results = (
            calculate_mean_lab_ita(
                illumination_corrected,
                cropped_mask
            )
        )

        # ----------------------------------------------------
        # STEP 2
        #
        # Hair removal after illumination correction
        # ----------------------------------------------------

        hair_removed_image, hair_mask = (
            remove_hair(
                illumination_corrected
            )
        )

        # ----------------------------------------------------
        # Restrict hair removal to current ROI
        # ----------------------------------------------------

        hair_mask = cv2.bitwise_and(
            hair_mask,
            cropped_mask
        )

        # ----------------------------------------------------
        # Reconstruct ROI-only hair removal
        # ----------------------------------------------------

        if np.any(
            hair_mask > 0
        ):

            final_image = cv2.inpaint(
                illumination_corrected,
                hair_mask,
                INPAINT_RADIUS,
                cv2.INPAINT_TELEA
            )

        else:

            final_image = (
                illumination_corrected.copy()
            )

        # ----------------------------------------------------
        # ITA after hair removal
        # ----------------------------------------------------

        hair_results = (
            calculate_mean_lab_ita(
                final_image,
                cropped_mask
            )
        )

        if index == 0:
            region_name = (
                "ROI 1 — Lesion"
            )
        elif index == 1:
            region_name = (
                "ROI 2 — Closed Lesion Region"
            )
        elif index == 2:
            region_name = (
                "ROI 3 — Healthy Region"
            )
        else:
            region_name = (
                f"ROI {index + 1}"
            )

        result = {
            "Region": region_name,
            "Original ROI File": roi_filename,
            "ROI Type": get_roi_type(roi),
            "Pixel Count":
                illumination_results["Pixel Count"],

            # ------------------------------------------------
            # Illumination correction
            # ------------------------------------------------

            "Illumination Mean L*":
                illumination_results["Mean L*"],

            "Illumination Mean a*":
                illumination_results["Mean a*"],

            "Illumination Mean b*":
                illumination_results["Mean b*"],

            "Illumination ITA":
                illumination_results["ITA"],

            # ------------------------------------------------
            # Hair removal
            # ------------------------------------------------

            "Hair Removal Mean L*":
                hair_results["Mean L*"],

            "Hair Removal Mean a*":
                hair_results["Mean a*"],

            "Hair Removal Mean b*":
                hair_results["Mean b*"],

            "Hair Removal ITA":
                hair_results["ITA"]
        }

        results.append(
            result
        )

        processed_rois.append({
            "region": region_name,
            "roi_file": roi_filename,
            "mask": cropped_mask,
            "illumination_corrected":
                illumination_corrected,
            "hair_mask":
                hair_mask,
            "final_image":
                final_image
        })

    return {
        "experiment": "E",
        "p": p,
        "results": results,
        "processed_rois": processed_rois
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def analyze_all_experiments(
    image_bgr,
    roi_zip,
    p=6
):
    """
    Run Experiments A-E.

    This function is provided for future integration
    into the Streamlit Research Summary.

    Each experiment remains scientifically independent.
    """

    return {

        "Experiment A":
            analyze_experiment_a(
                image_bgr,
                roi_zip,
                p=p
            ),

        "Experiment B":
            analyze_experiment_b(
                image_bgr,
                roi_zip,
                p=p
            ),

        "Experiment C":
            analyze_experiment_c(
                image_bgr,
                roi_zip,
                p=p
            ),

        "Experiment D":
            analyze_experiment_d(
                image_bgr,
                roi_zip
            ),

        "Experiment E":
            analyze_experiment_e(
                image_bgr,
                roi_zip,
                p=p
            )
    }