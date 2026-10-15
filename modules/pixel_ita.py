"""
Pixel-wise ITA analysis with preprocessing and a 3D ITA surface.

Ported from BioColorScan: hair removal (simple and experimental D/E),
Shades-of-Gray illumination correction, pixel-wise ITA map, histogram,
smoothed map, 3D surface, CIELAB means and downloads.
"""

import hashlib
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tifffile
from PIL import Image


# Experimental D/E hair-removal parameters (from BioColorScan)
ASPECT_RATIO_THRESHOLD = 1.4
KERNEL_SIZES = [9, 13]
ANGLES = list(range(0, 180, 15))
MAX_AREA_FRACTION = 0.03
MIN_AREA = 4


def load_image(uploaded_file):
    """Load JPG/JPEG/PNG/BMP/TIF/TIFF as RGB uint8."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith((".tif", ".tiff")):
        img = tifffile.imread(uploaded_file)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported TIFF shape: {img.shape}")

        if img.dtype != np.uint8:
            img = cv2.normalize(
                img, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

        return img

    return np.array(
        Image.open(uploaded_file).convert("RGB")
    ).astype(np.uint8)


def image_to_png_bytes(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def shades_of_gray(image_bgr, p=6):
    img_float = image_bgr.astype(np.float32)

    img_power = np.power(img_float, p)

    illum = np.power(
        np.mean(img_power, axis=(0, 1)),
        1.0 / p
    )

    norm = np.linalg.norm(illum)
    if norm == 0:
        return image_bgr.copy()

    illum_norm = (
        illum / norm * np.sqrt(3)
    )

    corrected = img_float / illum_norm

    return np.clip(
        corrected, 0, 255
    ).astype(np.uint8)


def make_line_kernel(length, angle_deg):
    k = np.zeros(
        (length, length),
        np.uint8
    )

    k[length // 2, :] = 1

    M = cv2.getRotationMatrix2D(
        (length / 2, length / 2),
        angle_deg,
        1
    )

    return cv2.warpAffine(
        k,
        M,
        (length, length)
    )


def experimental_hair_removal(image_rgb):
    """
    Experimental D/E hair removal:
    Blackhat + multiple line orientations + adaptive threshold +
    contour filtering + dilation + Telea inpainting.
    """
    img = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    img_area = gray.shape[0] * gray.shape[1]

    combined = np.zeros_like(gray)

    for klen in KERNEL_SIZES:
        for angle in ANGLES:
            kernel = make_line_kernel(
                klen,
                angle
            )

            bh = cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            combined = cv2.max(
                combined,
                bh
            )

    mask_adaptive = cv2.adaptiveThreshold(
        combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        -2
    )

    contours, _ = cv2.findContours(
        mask_adaptive,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hair_mask = np.zeros_like(
        mask_adaptive
    )

    for c in contours:
        area = cv2.contourArea(c)

        if (
            area < MIN_AREA
            or area > img_area * MAX_AREA_FRACTION
        ):
            continue

        x, y, w, h = cv2.boundingRect(c)

        aspect_ratio = (
            max(w, h) /
            (min(w, h) + 1e-5)
        )

        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
            cv2.drawContours(
                hair_mask,
                [c],
                -1,
                255,
                -1
            )

    hair_mask = cv2.dilate(
        hair_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    white_pct = (
        (hair_mask > 0).sum()
        / hair_mask.size
        * 100
    )

    result = cv2.inpaint(
        img,
        hair_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    return (
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        hair_mask,
        float(white_pct)
    )


def simple_hair_removal(image_rgb):
    """Hairremoval.py: 17x17 blackhat + threshold 10 + Telea."""
    img = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    result = cv2.inpaint(
        img,
        mask,
        1,
        cv2.INPAINT_TELEA
    )

    return (
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        mask
    )


def calculate_pixel_ita(image_rgb):
    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    L = lab[:, :, 0] * 100.0 / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    epsilon = 1e-6

    ita_map = np.degrees(
        np.arctan(
            (L - 50.0) /
            (b + epsilon)
        )
    )

    return L, a, b, ita_map


def create_ita_color_map(ita_map):
    ita_display = cv2.normalize(
        ita_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    ita_color = cv2.applyColorMap(
        ita_display,
        cv2.COLORMAP_VIRIDIS
    )

    return cv2.cvtColor(
        ita_color,
        cv2.COLOR_BGR2RGB
    )


# ------------------------------------------------------------------
# Streamlit section
# ------------------------------------------------------------------

def _compute(image_rgb, use_illum, use_simple, use_exp):
    processed = image_rgb.copy()
    hair_mask = None
    hair_pct = None

    if use_illum:
        bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        bgr = shades_of_gray(bgr, p=6)
        processed = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if use_simple:
        processed, hair_mask = simple_hair_removal(processed)

    if use_exp:
        processed, hair_mask, hair_pct = experimental_hair_removal(processed)

    L, a, b, ita_map = calculate_pixel_ita(processed)

    npy = io.BytesIO()
    np.save(npy, ita_map)

    csv = io.StringIO()
    np.savetxt(csv, ita_map, delimiter=",", fmt="%.4f")

    ita_color = create_ita_color_map(ita_map)

    return {
        "processed": processed,
        "hair_mask": hair_mask,
        "hair_pct": hair_pct,
        "L": float(np.mean(L)),
        "a": float(np.mean(a)),
        "b": float(np.mean(b)),
        "ita_map": ita_map,
        "ita_color": ita_color,
        "ita_smooth": cv2.GaussianBlur(ita_map, (11, 11), 0),
        "npy": npy.getvalue(),
        "csv": csv.getvalue(),
        "png": image_to_png_bytes(ita_color),
    }


def render_pixelwise_ita(image_rgb):
    st.markdown(
        "Choose optional preprocessing, then calculate the pixel-wise ITA map "
        "(ITA = arctan((L* − 50) / b*) for every pixel), its statistics, "
        "histogram, smoothed map and a 3D surface."
    )

    c1, c2, c3 = st.columns(3)
    use_simple = c1.checkbox("Simple hair removal", value=False, key="pita_simple")
    use_exp = c2.checkbox("Experimental D/E hair removal", value=False, key="pita_exp")
    use_illum = c3.checkbox("Shades-of-Gray illumination correction", value=False, key="pita_illum")

    key = (
        hashlib.md5(image_rgb.tobytes()).hexdigest(),
        use_simple, use_exp, use_illum,
    )

    if st.button("🔬 Calculate Pixel-wise ITA", type="primary", key="pita_button"):
        with st.spinner("Calculating pixel-wise ITA..."):
            st.session_state["pita"] = (key, _compute(image_rgb, use_illum, use_simple, use_exp))

    stored = st.session_state.get("pita")
    if not stored or stored[0] != key:
        st.info("Press the button to calculate (it recalculates if you change the image or the options).")
        return

    r = stored[1]
    ita_map = r["ita_map"]

    if use_illum or use_simple or use_exp:
        p1, p2 = st.columns(2)
        p1.image(r["processed"], caption="Processed image", use_container_width=True)
        if r["hair_mask"] is not None:
            p2.image(r["hair_mask"], caption="Hair mask", use_container_width=True)
        if r["hair_pct"] is not None:
            st.metric("Detected hair pixels", f"{r['hair_pct']:.2f}%")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Minimum", f"{np.min(ita_map):.2f}°")
    m2.metric("Maximum", f"{np.max(ita_map):.2f}°")
    m3.metric("Mean", f"{np.mean(ita_map):.2f}°")
    m4.metric("Median", f"{np.median(ita_map):.2f}°")
    m5.metric("Std. Dev.", f"{np.std(ita_map):.2f}°")

    # Primary 2D visualization: one map only.
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ita_map, cmap="viridis")
    ax.set_title("Pixel-wise ITA heat map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="ITA (degrees)")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # One distribution plot only.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    finite_values = ita_map[np.isfinite(ita_map)]
    ax.hist(finite_values, bins=60, edgecolor="black", linewidth=0.35)
    ax.set_xlabel("ITA (degrees)")
    ax.set_ylabel("Number of pixels")
    ax.set_title("Pixel-wise ITA distribution")
    ax.grid(True, alpha=0.22)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Additional visualizations are available only on demand, not repeated in the main view.
    with st.expander("Optional: smoothed ITA map and 3D surface", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(r["ita_smooth"], cmap="viridis")
        ax.set_title("Smoothed ITA map (optional view)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="ITA (degrees)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        scale = min(1.0, 300 / max(ita_map.shape))
        small = (
            cv2.resize(ita_map, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1 else ita_map
        )
        rows, cols = small.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(X, Y, small, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        ax.set_zlabel("ITA (degrees)")
        ax.set_title("3D pixel-wise ITA surface")
        fig.colorbar(surface, shrink=0.6, label="ITA (degrees)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("CIELAB information")
    k1, k2, k3 = st.columns(3)
    k1.metric("Mean L*", f"{r['L']:.2f}")
    k2.metric("Mean a*", f"{r['a']:.2f}")
    k3.metric("Mean b*", f"{r['b']:.2f}")

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ ITA map (.npy)", r["npy"], "ITA_Map.npy",
                       "application/octet-stream", key="pita_npy")
    d2.download_button("⬇ ITA data (.csv)", r["csv"], "ITA_Map.csv",
                       "text/csv", key="pita_csv")
    d3.download_button("⬇ False-colour ITA (.png)", r["png"], "ITA_Color_Map.png",
                       "image/png", key="pita_png")
