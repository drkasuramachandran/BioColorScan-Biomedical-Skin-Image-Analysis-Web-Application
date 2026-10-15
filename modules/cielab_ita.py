import cv2
import numpy as np


def calculate_ita(image):

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None:
        raise ValueError(
            "Input image is empty."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise TypeError(
            "Input image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Input image contains no pixels."
        )

    # --------------------------------------------------------
    # Convert image to uint8
    # --------------------------------------------------------

    if image.dtype != np.uint8:

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    # --------------------------------------------------------
    # RGB → CIELAB
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # --------------------------------------------------------
    # Extract CIELAB components
    # --------------------------------------------------------

    L = (
        lab[:, :, 0]
        * 100.0
        / 255.0
    )

    a = (
        lab[:, :, 1]
        - 128.0
    )

    b = (
        lab[:, :, 2]
        - 128.0
    )

    # --------------------------------------------------------
    # Individual Typology Angle (ITA)
    #
    # ITA = arctan((L* - 50) / b*)
    # --------------------------------------------------------

    epsilon = 1e-6

    ita_map = np.degrees(
        np.arctan(
            (L - 50.0)
            / (b + epsilon)
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    results = {

        "Minimum ITA":
            float(np.min(ita_map)),

        "Maximum ITA":
            float(np.max(ita_map)),

        "Mean ITA":
            float(np.mean(ita_map)),

        "Median ITA":
            float(np.median(ita_map)),

        "Standard Deviation":
            float(np.std(ita_map))
    }

    return results, ita_map


def create_ita_visualizations(
    ita_map
):

    # --------------------------------------------------------
    # Normalize ITA for display
    # --------------------------------------------------------

    ita_display = cv2.normalize(
        ita_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --------------------------------------------------------
    # False-colour ITA map
    # --------------------------------------------------------

    ita_color = cv2.applyColorMap(
        ita_display,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # Smoothed ITA map
    # --------------------------------------------------------

    ita_smooth = cv2.GaussianBlur(
        ita_map,
        (11, 11),
        0
    )

    return (
        ita_display,
        ita_color,
        ita_smooth
    )


def analyze_cielab_ita(
    image
):

    results, ita_map = calculate_ita(
        image
    )

    (
        ita_display,
        ita_color,
        ita_smooth
    ) = create_ita_visualizations(
        ita_map
    )

    return {

        "results":
            results,

        "ita_map":
            ita_map,

        "ita_display":
            ita_display,

        "ita_color":
            ita_color,

        "ita_smooth":
            ita_smooth
    }