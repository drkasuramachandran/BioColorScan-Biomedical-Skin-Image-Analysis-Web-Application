# ============================================================
# THREE-REGION IMAGEJ ROI PIXEL-WISE ITA ANALYSIS
# ============================================================
#
# Scientific methodology
#
# ROI 1              = Lesion
# ROI 2 - ROI 1      = Surrounding healthy skin
# ROI 3 - ROI 2      = Outer healthy skin
#
# The subtraction is performed at the MASK / PIXEL level.
#
# ITA is then calculated pixel-wise for each derived region.
#
# ITA = atan2(L* - 50, b*) x 180 / pi
#
# No:
#   - illumination correction
#   - hair removal
#   - colour normalization
#   - CNN processing
#
# ============================================================

import io
import zipfile

import cv2
import numpy as np


# ============================================================
# CIELAB CONVERSION
# ============================================================

def convert_to_cielab(image_bgr):
    """
    Convert BGR image to conventional CIELAB ranges.

    Returns:
        L : approximately 0 to 100
        a : approximately -128 to +127
        b : approximately -128 to +127
    """

    if image_bgr is None:
        raise ValueError("Input image is empty.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image_bgr.size == 0:
        raise ValueError("Input image contains no pixels.")

    if image_bgr.dtype != np.uint8:
        image_bgr = cv2.normalize(
            image_bgr,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    L = lab[:, :, 0] * (100.0 / 255.0)

    a = lab[:, :, 1] - 128.0

    b = lab[:, :, 2] - 128.0

    return L, a, b


# ============================================================
# IMAGEJ ROI TYPE
# ============================================================

def get_roi_type(roi):
    """
    Return a readable ImageJ ROI type.
    """

    try:
        return str(roi.roitype)
    except Exception:
        return "Unknown"


# ============================================================
# IMAGEJ ROI TO MASK
# ============================================================

def roi_to_mask(roi, image_shape):
    """
    Convert an ImageJ ROI into a full-size binary mask.

    Supports common ImageJ ROI types:
        - Polygon
        - Freehand
        - Rectangle
        - Oval
        - Coordinate-based ROIs
    """

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
                        [points.reshape((-1, 1, 2))],
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
    """
    Read ImageJ ROI files from a ZIP archive.

    ROI files are sorted by filename.

    Returns:
        list of (filename, ImagejRoi)
    """

    if roi_zip is None:
        raise ValueError(
            "ROI ZIP file is empty."
        )

    # --------------------------------------------------------
    # Obtain ZIP bytes
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
    # Open ZIP
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

        for roi_filename in roi_files:

            roi_bytes = zip_file.read(
                roi_filename
            )

            try:
                from roifile import ImagejRoi
            except ImportError as exc:
                raise ImportError(
                    "The 'roifile' package is required for ImageJ ROI analysis. "
                    "Install it with: python -m pip install roifile"
                ) from exc

            roi = ImagejRoi.frombytes(
                roi_bytes
            )

            rois.append(
                (
                    roi_filename,
                    roi
                )
            )

    return rois


# ============================================================
# PIXEL-WISE ITA
# ============================================================

def calculate_pixelwise_ita(
    L,
    b,
    mask
):
    """
    Calculate pixel-wise ITA statistics for a region.
    """

    region_pixels = mask > 0

    if not np.any(region_pixels):

        return {
            "Valid Pixels": 0,
            "Minimum ITA": np.nan,
            "Maximum ITA": np.nan,
            "Mean ITA": np.nan,
            "Median ITA": np.nan,
            "Standard Deviation": np.nan
        }, np.array([], dtype=np.float32)

    L_region = L[region_pixels]

    b_region = b[region_pixels]

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    valid = (
        np.isfinite(L_region)
        &
        np.isfinite(b_region)
        &
        (np.abs(b_region) > 1e-12)
    )

    L_region = L_region[valid]

    b_region = b_region[valid]

    if len(L_region) == 0:

        return {
            "Valid Pixels": 0,
            "Minimum ITA": np.nan,
            "Maximum ITA": np.nan,
            "Mean ITA": np.nan,
            "Median ITA": np.nan,
            "Standard Deviation": np.nan
        }, np.array([], dtype=np.float32)

    # --------------------------------------------------------
    # PIXEL-WISE ITA
    # --------------------------------------------------------

    ita_values = np.degrees(
        np.arctan2(
            L_region - 50.0,
            b_region
        )
    )

    ita_values = ita_values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    results = {

        "Valid Pixels":
            int(len(ita_values)),

        "Minimum ITA":
            float(np.min(ita_values)),

        "Maximum ITA":
            float(np.max(ita_values)),

        "Mean ITA":
            float(np.mean(ita_values)),

        "Median ITA":
            float(np.median(ita_values)),

        "Standard Deviation":
            float(np.std(ita_values))
    }

    return results, ita_values


# ============================================================
# BASELINE ROI PIXEL-WISE ITA ANALYSIS
# ============================================================

def analyze_roi_pixelwise_ita(
    image_bgr,
    roi_zip
):
    """
    Baseline pixel-wise ITA analysis for each original ImageJ ROI.

    Unlike the three-region analysis, this function does not perform
    ROI subtraction. Each uploaded ROI is analysed independently.

    Returns a dictionary containing the ROI list, masks, results and
    pixel-wise ITA values.
    """

    if image_bgr is None:
        raise ValueError("Input image is empty.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image_bgr.size == 0:
        raise ValueError("Input image contains no pixels.")

    roi_list = load_imagej_rois(roi_zip)

    if len(roi_list) == 0:
        raise ValueError("At least one ImageJ ROI is required.")

    L, a, b = convert_to_cielab(image_bgr)

    results = []
    masks = []
    ita_values = []

    for index, (roi_name, roi) in enumerate(roi_list, start=1):
        mask = roi_to_mask(roi, image_bgr.shape)
        stats, values = calculate_pixelwise_ita(L, b, mask)

        results.append({
            "ROI": f"ROI {index}",
            "Original ROI": roi_name,
            "ROI Type": get_roi_type(roi),
            **stats
        })
        masks.append(mask)
        ita_values.append(values)

    return {
        "roi_list": roi_list,
        "masks": masks,
        "results": results,
        "ita_values": ita_values
    }


# ============================================================
# THREE-REGION ANALYSIS
# ============================================================

def analyze_three_region_ita(
    image_bgr,
    roi_zip
):
    """
    Perform the three-region ImageJ ROI analysis.

    ROI definitions:

        ROI 1
        -----
        Lesion

        ROI 2 - ROI 1
        ------------
        Surrounding healthy skin

        ROI 3 - ROI 2
        ------------
        Outer healthy skin

    ROI subtraction is performed at the binary mask level.

    Returns:
        Dictionary containing masks, results and ITA values.
    """

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image_bgr is None:
        raise ValueError(
            "Input image is empty."
        )

    if not isinstance(
        image_bgr,
        np.ndarray
    ):
        raise TypeError(
            "Input image must be a NumPy array."
        )

    if image_bgr.size == 0:
        raise ValueError(
            "Input image contains no pixels."
        )

    # --------------------------------------------------------
    # Load ROIs
    # --------------------------------------------------------

    roi_list = load_imagej_rois(
        roi_zip
    )

    if len(roi_list) < 3:

        raise ValueError(
            "At least three ImageJ ROIs are required."
        )

    if len(roi_list) > 3:

        # For the three-region experiment we use
        # the first three ROIs after filename sorting.
        roi_list = roi_list[:3]

    # --------------------------------------------------------
    # Create masks
    # --------------------------------------------------------

    roi1_name, roi1 = roi_list[0]

    roi2_name, roi2 = roi_list[1]

    roi3_name, roi3 = roi_list[2]

    roi1_mask = roi_to_mask(
        roi1,
        image_bgr.shape
    )

    roi2_mask = roi_to_mask(
        roi2,
        image_bgr.shape
    )

    roi3_mask = roi_to_mask(
        roi3,
        image_bgr.shape
    )

    # --------------------------------------------------------
    # Convert to boolean
    # --------------------------------------------------------

    mask1 = roi1_mask > 0

    mask2 = roi2_mask > 0

    mask3 = roi3_mask > 0

    # --------------------------------------------------------
    # DERIVED THREE REGIONS
    #
    # Region 1:
    #     ROI 1
    #
    # Region 2:
    #     ROI 2 - ROI 1
    #
    # Region 3:
    #     ROI 3 - ROI 2
    # --------------------------------------------------------

    lesion_mask = mask1

    surrounding_healthy_mask = (
        mask2
        &
        (~mask1)
    )

    outer_healthy_mask = (
        mask3
        &
        (~mask2)
    )

    # Convert back to uint8 masks
    lesion_mask_uint8 = (
        lesion_mask.astype(np.uint8) * 255
    )

    surrounding_healthy_mask_uint8 = (
        surrounding_healthy_mask.astype(np.uint8) * 255
    )

    outer_healthy_mask_uint8 = (
        outer_healthy_mask.astype(np.uint8) * 255
    )

    # --------------------------------------------------------
    # CIELAB
    # --------------------------------------------------------

    L, a, b = convert_to_cielab(
        image_bgr
    )

    # --------------------------------------------------------
    # CALCULATE ITA FOR EACH REGION
    # --------------------------------------------------------

    lesion_results, lesion_ita = (
        calculate_pixelwise_ita(
            L,
            b,
            lesion_mask_uint8
        )
    )

    surrounding_results, surrounding_ita = (
        calculate_pixelwise_ita(
            L,
            b,
            surrounding_healthy_mask_uint8
        )
    )

    outer_results, outer_ita = (
        calculate_pixelwise_ita(
            L,
            b,
            outer_healthy_mask_uint8
        )
    )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results = [

        {
            "Region": "ROI 1",
            "Meaning": "Lesion",
            "Original ROI": roi1_name,
            "ROI Type": get_roi_type(roi1),
            **lesion_results
        },

        {
            "Region": "ROI 2 - ROI 1",
            "Meaning": "Surrounding Healthy Skin",
            "Original ROI": (
                f"{roi2_name} - {roi1_name}"
            ),
            "ROI Type": get_roi_type(roi2),
            **surrounding_results
        },

        {
            "Region": "ROI 3 - ROI 2",
            "Meaning": "Outer Healthy Skin",
            "Original ROI": (
                f"{roi3_name} - {roi2_name}"
            ),
            "ROI Type": get_roi_type(roi3),
            **outer_results
        }
    ]

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "roi_list": roi_list,

        "original_masks": [
            roi1_mask,
            roi2_mask,
            roi3_mask
        ],

        "derived_masks": [

            lesion_mask_uint8,

            surrounding_healthy_mask_uint8,

            outer_healthy_mask_uint8
        ],

        "results": results,

        "ita_values": [

            lesion_ita,

            surrounding_ita,

            outer_ita
        ],

        "region_names": [

            "ROI 1 — Lesion",

            "ROI 2 − ROI 1 — Surrounding Healthy Skin",

            "ROI 3 − ROI 2 — Outer Healthy Skin"
        ]
    }


# ============================================================
# CREATE ITA MAP
# ============================================================

def create_region_ita_map(
    image_bgr,
    mask
):
    """
    Create a pixel-wise ITA map for a selected region.

    Pixels outside the region are NaN.
    """

    L, a, b = convert_to_cielab(
        image_bgr
    )

    ita_map = np.full(
        L.shape,
        np.nan,
        dtype=np.float32
    )

    region_pixels = mask > 0

    valid = (
        region_pixels
        &
        np.isfinite(L)
        &
        np.isfinite(b)
        &
        (np.abs(b) > 1e-12)
    )

    ita_map[valid] = np.degrees(
        np.arctan2(
            L[valid] - 50.0,
            b[valid]
        )
    )

    return ita_map


# ============================================================
# ITA DISPLAY
# ============================================================

def create_ita_display(
    ita_map
):
    """
    Normalize an ITA map for visualization only.

    The original ITA values are not changed.
    """

    display = np.zeros(
        ita_map.shape,
        dtype=np.uint8
    )

    valid = np.isfinite(
        ita_map
    )

    if not np.any(valid):

        return display

    values = ita_map[valid]

    min_value = np.min(values)

    max_value = np.max(values)

    if max_value > min_value:

        normalized = (
            (ita_map - min_value)
            /
            (max_value - min_value)
            *
            255.0
        )

        display[valid] = np.clip(
            normalized[valid],
            0,
            255
        ).astype(np.uint8)

    else:

        display[valid] = 128

    return display