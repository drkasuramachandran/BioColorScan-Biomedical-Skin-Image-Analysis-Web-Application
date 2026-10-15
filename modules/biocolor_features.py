"""
BioColorScan feature module
---------------------------
Functions copied unchanged from BioColorScan_Final.py so that the merged
application keeps every BioColorScan feature:

  * common image helpers (TIFF-aware loader, download helpers)
  * Shades-of-Gray illumination correction
  * simple + experimental (Experiment D/E) hair removal
  * ABCDE-style analysis with pixel-size aware diameter and evolution score
  * pixel-wise ITA with histogram, smoothed map, 3D surface, CIELAB means
    and .npy / .csv / .png downloads
  * Experiments A-E (image pipelines + L*a*b* TIFF-channel ITA)
"""

import io
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import tifffile
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# COMMON IMAGE FUNCTIONS
# ============================================================

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


def image_to_jpg_bytes(image_rgb, quality=95):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(
        buffer, format="JPEG", quality=quality
    )
    buffer.seek(0)
    return buffer.getvalue()


def image_to_png_bytes(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def fig_to_png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=150
    )
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SHADES OF GRAY - EXPERIMENTS B, C, E
# ============================================================

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


def experiment_b_pipeline(image_rgb):
    """Experiment B: resize to 224x224, then Shades-of-Gray p=6."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(
        image_bgr,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    sog_bgr = shades_of_gray(resized, p=6)

    return (
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(sog_bgr, cv2.COLOR_BGR2RGB)
    )


def experiment_c_pipeline(image_rgb):
    """
    Experiment C:
    Shades-of-Gray p=6 -> LAB -> L-channel stretching ->
    reconstruct image.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    color_corrected = shades_of_gray(
        image_bgr,
        p=6
    )

    lab = cv2.cvtColor(
        color_corrected,
        cv2.COLOR_BGR2LAB
    )

    l_ch, a_ch, b_ch = cv2.split(lab)

    l_stretched = cv2.normalize(
        l_ch,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    reconstructed = cv2.cvtColor(
        cv2.merge([
            l_stretched,
            a_ch,
            b_ch
        ]),
        cv2.COLOR_LAB2BGR
    )

    return (
        cv2.cvtColor(color_corrected, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)
    )


# ============================================================
# EXPERIMENT D HAIR REMOVAL
# ============================================================

ASPECT_RATIO_THRESHOLD = 1.4
KERNEL_SIZES = [9, 13]
ANGLES = list(range(0, 180, 15))
MAX_AREA_FRACTION = 0.03
MIN_AREA = 4


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


# ============================================================
# ORIGINAL SIMPLE HAIR REMOVAL
# ============================================================

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


# ============================================================
# ITA FROM RGB - ORIGINAL BIOSCAN METHOD
# ============================================================

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


# ============================================================
# MEAN CIELAB ITA - EXPERIMENTS A-E
# ============================================================

def calculate_mean_lab_ita(L, a, b):
    """
    Experimental A-E ITA calculation:
    mean L*, mean a*, mean b*, then
    ITA = degrees(arctan((L_mean - 50) / b_mean)).
    """
    L_mean = float(np.mean(L))
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))

    if abs(b_mean) < 1e-12:
        ita = float("nan")
    else:
        ita = float(
            np.degrees(
                np.arctan(
                    (L_mean - 50.0) /
                    b_mean
                )
            )
        )

    return L_mean, a_mean, b_mean, ita


# ============================================================
# ABCDE ANALYSIS
# ============================================================

def create_lesion_mask(image_rgb):
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


def calculate_abcde_features(
    image_rgb,
    pixel_size_mm=0.05
):
    mask = create_lesion_mask(image_rgb)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No lesion contour was detected."
        )

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(
        cnt,
        True
    )

    if area <= 0:
        raise ValueError(
            "Detected lesion area is zero."
        )

    circularity = (
        4 * np.pi * area /
        (perimeter ** 2)
        if perimeter > 0
        else 0.0
    )

    pixels = image_rgb[mask > 0]

    if pixels.size == 0:
        raise ValueError(
            "No pixels were found inside lesion mask."
        )

    std_rgb = np.std(
        pixels,
        axis=0
    )

    color_variation = float(
        np.mean(std_rgb)
    )

    x, y, width, height = cv2.boundingRect(cnt)

    diameter = (
        max(width, height) *
        pixel_size_mm
    )

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

    mask_difference = np.sum(
        np.abs(
            left.astype(int) -
            right.astype(int)
        )
    )

    mask_area = np.sum(
        mask > 0
    )

    asymmetry = (
        float(mask_difference / mask_area)
        if mask_area > 0
        else 0.0
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

    return {
        "mask": mask,
        "contour": cnt,
        "area_pixels": float(area),
        "perimeter_pixels": float(perimeter),
        "asymmetry": asymmetry,
        "circularity": float(circularity),
        "color_variation": color_variation,
        "diameter_mm": float(diameter),
        "score_without_evolution": score
    }


def calculate_evolution_score(old_rgb, new_rgb):
    if old_rgb.shape[:2] != new_rgb.shape[:2]:
        new_rgb = cv2.resize(
            new_rgb,
            (old_rgb.shape[1], old_rgb.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    return float(
        np.mean(
            np.abs(
                old_rgb.astype(float) -
                new_rgb.astype(float)
            )
        )
    )



def render_preprocessing(image_rgb):
    """Preprocessing controls. Returns (processed_rgb, hair_mask)."""

    p1, p2 = st.columns(2)

    with p1:
        use_simple_hair = st.checkbox(
            "Simple hair removal",
            value=False,
            key="main_simple_hair"
        )

    with p2:
        use_exp_hair = st.checkbox(
            "Experimental D/E hair removal",
            value=True,
            key="main_exp_hair"
        )

    use_illumination = st.checkbox(
        "Shades-of-Gray illumination correction",
        value=False,
        key="main_illumination"
    )

    processed = image_rgb.copy()
    hair_mask = None

    if use_illumination:
        bgr = cv2.cvtColor(
            processed,
            cv2.COLOR_RGB2BGR
        )
        bgr = shades_of_gray(
            bgr,
            p=6
        )
        processed = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

    if use_simple_hair:
        processed, hair_mask = (
            simple_hair_removal(processed)
        )

    if use_exp_hair:
        processed, hair_mask, hair_pct = (
            experimental_hair_removal(
                processed
            )
        )
        st.metric(
            "Detected hair pixels",
            f"{hair_pct:.2f}%"
        )

    if (
        use_illumination
        or use_simple_hair
        or use_exp_hair
    ):
        c1, c2 = st.columns(2)

        with c1:
            st.image(
                processed,
                caption="Processed Image",
                use_container_width=True
            )

        if hair_mask is not None:
            with c2:
                st.image(
                    hair_mask,
                    caption="Hair Mask",
                    use_container_width=True
                )
    return processed, hair_mask


def render_abcde_biocolor(processed, pixel_size_mm=0.05):
    try:
        abcde = calculate_abcde_features(
            processed,
            pixel_size_mm
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Asymmetry",
                f"{abcde['asymmetry']:.4f}"
            )

        with c2:
            st.metric(
                "Circularity",
                f"{abcde['circularity']:.4f}"
            )

        with c3:
            st.metric(
                "Color Variation",
                f"{abcde['color_variation']:.2f}"
            )

        with c4:
            st.metric(
                "Diameter",
                f"{abcde['diameter_mm']:.2f} mm"
            )

        st.image(
            abcde["mask"],
            caption="Lesion Mask",
            use_container_width=True
        )

        old_file = st.file_uploader(
            "Optional previous image for Evolution",
            type=[
                "jpg", "jpeg", "png", "bmp",
                "tif", "tiff"
            ],
            key="main_old"
        )

        evolution_score = None
        evolution_flag = 0

        if old_file is not None:
            old_rgb = load_image(
                old_file
            )

            evolution_score = (
                calculate_evolution_score(
                    old_rgb,
                    processed
                )
            )

            st.metric(
                "Evolution Score",
                f"{evolution_score:.4f}"
            )

            if evolution_score > 20:
                evolution_flag = 1

        final_score = (
            abcde["score_without_evolution"]
            + evolution_flag
        )

        st.metric(
            "ABCDE Score",
            f"{final_score} / 5"
        )

    except Exception as exc:
        st.error(
            f"ABCDE analysis failed: {exc}"
        )


def render_pixelwise_ita_biocolor(processed):

    if st.button(
        "🔬 Calculate Pixel-wise ITA",
        type="primary",
        key="main_ita_button"
    ):
        L, a, b, ita_map = (
            calculate_pixel_ita(
                processed
            )
        )

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        with c1:
            st.metric(
                "Minimum",
                f"{np.min(ita_map):.2f}°"
            )
        with c2:
            st.metric(
                "Maximum",
                f"{np.max(ita_map):.2f}°"
            )
        with c3:
            st.metric(
                "Mean",
                f"{np.mean(ita_map):.2f}°"
            )
        with c4:
            st.metric(
                "Median",
                f"{np.median(ita_map):.2f}°"
            )
        with c5:
            st.metric(
                "Std. Dev.",
                f"{np.std(ita_map):.2f}°"
            )

        ita_color = create_ita_color_map(
            ita_map
        )

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(
                figsize=(7, 6)
            )
            im = ax.imshow(
                ita_map,
                cmap="viridis"
            )
            ax.set_title(
                "Pixel-wise ITA Map"
            )
            ax.axis("off")
            fig.colorbar(
                im,
                ax=ax,
                label="ITA (Degrees)"
            )
            st.pyplot(
                fig,
                use_container_width=True
            )
            plt.close(fig)

        with c2:
            st.image(
                ita_color,
                caption="False Colour ITA Map",
                use_container_width=True
            )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )
        ax.hist(
            ita_map.flatten(),
            bins=100,
            edgecolor="black"
        )
        ax.set_xlabel(
            "ITA (Degrees)"
        )
        ax.set_ylabel(
            "Number of Pixels"
        )
        ax.set_title(
            "Histogram of Pixel-wise ITA"
        )
        ax.grid(True)
        st.pyplot(
            fig,
            use_container_width=True
        )
        plt.close(fig)

        ita_smooth = cv2.GaussianBlur(
            ita_map,
            (11, 11),
            0
        )

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )
        im = ax.imshow(
            ita_smooth,
            cmap="viridis"
        )
        ax.set_title(
            "Smoothed Pixel-wise ITA Map"
        )
        ax.axis("off")
        fig.colorbar(
            im,
            ax=ax,
            label="ITA (Degrees)"
        )
        st.pyplot(
            fig,
            use_container_width=True
        )
        plt.close(fig)

        st.subheader(
            "3D Pixel-wise ITA Surface"
        )

        max_dimension = 300
        scale = min(
            1.0,
            max_dimension /
            max(ita_map.shape)
        )

        if scale < 1:
            small = cv2.resize(
                ita_map,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            small = ita_map

        rows, cols = small.shape
        X, Y = np.meshgrid(
            np.arange(cols),
            np.arange(rows)
        )

        fig = plt.figure(
            figsize=(12, 8)
        )
        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        surface = ax.plot_surface(
            X,
            Y,
            small,
            cmap="viridis",
            linewidth=0,
            antialiased=True
        )

        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        ax.set_zlabel("ITA (Degrees)")
        ax.set_title(
            "3D Pixel-wise ITA Surface"
        )

        fig.colorbar(
            surface,
            shrink=0.6,
            label="ITA"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )
        plt.close(fig)

        st.subheader("CIELAB Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Mean L*",
                f"{np.mean(L):.2f}"
            )
        with c2:
            st.metric(
                "Mean a*",
                f"{np.mean(a):.2f}"
            )
        with c3:
            st.metric(
                "Mean b*",
                f"{np.mean(b):.2f}"
            )

        npy_buffer = io.BytesIO()
        np.save(
            npy_buffer,
            ita_map
        )
        npy_buffer.seek(0)

        csv_text = io.StringIO()
        np.savetxt(
            csv_text,
            ita_map,
            delimiter=",",
            fmt="%.4f"
        )

        st.download_button(
            "⬇ Download ITA Map (.npy)",
            npy_buffer.getvalue(),
            "ITA_Map.npy",
            "application/octet-stream",
            key="main_npy"
        )

        st.download_button(
            "⬇ Download ITA Data (.csv)",
            csv_text.getvalue(),
            "ITA_Map.csv",
            "text/csv",
            key="main_csv"
        )

        st.download_button(
            "⬇ Download False Colour ITA (.png)",
            image_to_png_bytes(ita_color),
            "ITA_Color_Map.png",
            "image/png",
            key="main_color_png"
        )



def rgb_to_cielab_channels(image_rgb):
    """Convert an RGB uint8 image to CIELAB arrays in standard units.

    L* is approximately 0..100; a* and b* use signed CIELAB units.
    OpenCV's float RGB conversion avoids the uint8 a/b offset encoding.
    """
    rgb_float = np.asarray(image_rgb, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def _channel_tiff_bytes(channel):
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, np.asarray(channel, dtype=np.float32))
    buffer.seek(0)
    return buffer.getvalue()


def _show_lab_ita_result(image_rgb, label, key):
    """Generate and show CIELAB channels and mean-channel ITA for one pipeline."""
    L, a, b = rgb_to_cielab_channels(image_rgb)
    L_mean, a_mean, b_mean, ita = calculate_mean_lab_ita(L, a, b)

    st.markdown(f"#### {label}: automatically generated CIELAB channels")
    previews = st.columns(3)
    channel_items = [("L* — Lightness", L), ("a* — Green–red", a), ("b* — Blue–yellow", b)]
    for col, (caption, channel) in zip(previews, channel_items):
        # Preview is normalized only for display; downloads retain actual CIELAB values.
        display = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        with col:
            st.image(display, caption=caption, use_container_width=True, clamp=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, title, value, suffix in [
        (m1, "Mean L*", L_mean, ""),
        (m2, "Mean a*", a_mean, ""),
        (m3, "Mean b*", b_mean, ""),
        (m4, "Mean-channel ITA", ita, "°"),
    ]:
        with col:
            st.metric(title, f"{value:.3f}{suffix}" if np.isfinite(value) else "N/A")

    d1, d2, d3 = st.columns(3)
    for col, name, channel, suffix in [
        (d1, "L", L, "L"), (d2, "a", a, "a"), (d3, "b", b, "b")
    ]:
        with col:
            st.download_button(
                f"Download {name}* channel TIFF",
                _channel_tiff_bytes(channel),
                f"{key}_{suffix}_channel.tif",
                "image/tiff",
                key=f"{key}_download_{suffix}",
                use_container_width=True,
            )
    return {"Experiment": label, "Mean L*": L_mean, "Mean a*": a_mean,
            "Mean b*": b_mean, "Mean-channel ITA (degrees)": ita}


def render_experiment_a(image_rgb):
    st.subheader("Experiment A — Original image → CIELAB → ITA")
    st.caption("Baseline: CIELAB channels are generated directly from the uploaded original image.")
    st.image(image_rgb, caption="Original image", use_container_width=True)
    return _show_lab_ita_result(image_rgb, "Experiment A — Original", "bio_exp_a")


def render_experiment_b(image_rgb):
    st.subheader("Experiment B — Resize → Shades-of-Gray (p=6) → CIELAB → ITA")
    resized, sog = experiment_b_pipeline(image_rgb)
    c1, c2 = st.columns(2)
    with c1:
        st.image(resized, caption="Resized image — 224 × 224", use_container_width=True)
    with c2:
        st.image(sog, caption="Shades-of-Gray corrected image — p=6", use_container_width=True)
    st.download_button("Download resized image", image_to_jpg_bytes(resized), "experiment_b_resized.jpg", "image/jpeg", key="bio_exp_b_resized")
    st.download_button("Download corrected image", image_to_jpg_bytes(sog), "experiment_b_shades_of_gray.jpg", "image/jpeg", key="bio_exp_b_sog")
    return _show_lab_ita_result(sog, "Experiment B — Shades-of-Gray", "bio_exp_b")


def render_experiment_c(image_rgb):
    st.subheader("Experiment C — Shades-of-Gray → L* stretching → CIELAB → ITA")
    sog, stretched = experiment_c_pipeline(image_rgb)
    c1, c2 = st.columns(2)
    with c1:
        st.image(sog, caption="Shades-of-Gray corrected — p=6", use_container_width=True)
    with c2:
        st.image(stretched, caption="L*-stretched reconstructed image", use_container_width=True)
    st.download_button("Download L*-stretched image", image_to_jpg_bytes(stretched), "experiment_c_l_stretched.jpg", "image/jpeg", key="bio_exp_c_image")
    return _show_lab_ita_result(stretched, "Experiment C — L*-stretched", "bio_exp_c")


def render_experiment_d(image_rgb):
    st.subheader("Experiment D — Hair removal → CIELAB → ITA")
    result, mask, hair_pct = experimental_hair_removal(image_rgb)
    c1, c2 = st.columns(2)
    with c1:
        st.image(result, caption="Hair-removed image", use_container_width=True)
    with c2:
        st.image(mask, caption="Detected hair mask", use_container_width=True)
    st.metric("Detected hair pixels", f"{hair_pct:.2f}%")
    st.download_button("Download hair-removed image", image_to_jpg_bytes(result), "experiment_d_hair_removed.jpg", "image/jpeg", key="bio_exp_d_image")
    return _show_lab_ita_result(result, "Experiment D — Hair removed", "bio_exp_d")


def render_experiment_e(image_rgb):
    st.subheader("Experiment E — Illumination correction → Hair removal → CIELAB → ITA")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    corrected = cv2.cvtColor(shades_of_gray(image_bgr, p=6), cv2.COLOR_BGR2RGB)
    result, mask, hair_pct = experimental_hair_removal(corrected)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(image_rgb, caption="Original", use_container_width=True)
    with c2:
        st.image(corrected, caption="Shades-of-Gray corrected — p=6", use_container_width=True)
    with c3:
        st.image(result, caption="After hair removal", use_container_width=True)
    st.image(mask, caption="Detected hair mask", use_container_width=True)
    st.metric("Detected hair pixels", f"{hair_pct:.2f}%")
    st.download_button("Download Experiment E processed image", image_to_jpg_bytes(result), "experiment_e_processed.jpg", "image/jpeg", key="bio_exp_e_image")
    return _show_lab_ita_result(result, "Experiment E — Corrected + hair removed", "bio_exp_e")


def render_experiments_a_to_e(image_rgb=None):
    """Run BioColorScan A-E from one shared original image upload.

    When called from the main app, the already-uploaded image is passed in,
    so this section never asks the user to upload L*/a*/b* TIFFs separately.
    """
    if image_rgb is None:
        uploaded = st.file_uploader(
            "Upload one original skin-lesion image for Experiments A–E",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="biocolor_shared_original_image",
        )
        if uploaded is None:
            st.info("Upload one original image to automatically generate the CIELAB channels and run Experiments A–E.")
            return
        try:
            image_rgb = load_image(uploaded)
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            return

    st.info(
        "One original image is reused across Experiments A–E. Each experiment "
        "generates its own L*, a*, and b* channels automatically after its "
        "specific preprocessing pipeline. TIFF channel downloads preserve "
        "the numerical CIELAB values; preview images are normalized for display."
    )
    tabs = st.tabs(["A · Original", "B · Resize + SoG", "C · L* stretch", "D · Hair removal", "E · SoG + hair removal"])
    results = []
    renderers = [render_experiment_a, render_experiment_b, render_experiment_c, render_experiment_d, render_experiment_e]
    for tab, renderer in zip(tabs, renderers):
        with tab:
            try:
                results.append(renderer(image_rgb))
            except Exception as exc:
                st.error(f"{renderer.__name__.replace('render_experiment_', 'Experiment ').upper()} failed: {exc}")
    if results:
        summary = pd.DataFrame(results)
        # Share the computed A–E results with Compare Experiments; display the
        # comparison table only once in the dedicated comparison section.
        st.session_state["biocolor_ae_results"] = summary.to_dict("records")
        st.success("Experiments A–E completed. Open ‘Compare Experiments A–E’ below for the single consolidated table and chart.")
        st.download_button(
            "Download A–E results CSV",
            summary.to_csv(index=False).encode("utf-8"),
            "biocolorscan_experiments_A_to_E_summary.csv",
            "text/csv",
            key="bio_exp_ae_summary_csv",
        )
