import os
import io
import zipfile
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import tifffile
import matplotlib.pyplot as plt
from PIL import Image

try:
    from roifile import ImagejRoi
    ROIFILE_AVAILABLE = True
except ImportError:
    ROIFILE_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BioColorScan | Biomedical Skin Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BIOSCAN VISUAL DESIGN
# ============================================================

st.markdown("""
<style>

/* ---------------------------------------------------------
   GLOBAL PAGE
--------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 188, 212, 0.12), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(124, 77, 255, 0.10), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(0, 150, 136, 0.08), transparent 35%),
        #f5f8fc;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b2545 0%,
            #123b63 45%,
            #0d5366 100%
        );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stNumberInput input {
    color: #102a43 !important;
    background: white !important;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #d8f3f7 !important;
}

/* ---------------------------------------------------------
   MAIN HERO
--------------------------------------------------------- */

.biocolorscan-hero {
    position: relative;
    overflow: hidden;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            #082f49 0%,
            #0b6477 45%,
            #1976a8 72%,
            #5145cd 100%
        );
    box-shadow: 0 15px 40px rgba(13, 71, 102, 0.20);
    color: white;
}

.biocolorscan-hero::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
}

.biocolorscan-hero::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: 180px;
    bottom: -120px;
    border-radius: 50%;
    background: rgba(0,229,255,0.12);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}

.hero-subtitle {
    font-size: 1.15rem;
    margin-top: 0.45rem;
    color: #dffaff;
}

.hero-description {
    max-width: 900px;
    margin-top: 1rem;
    line-height: 1.65;
    color: #edfaff;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1.3rem;
}

.hero-badge {
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    color: white;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ---------------------------------------------------------
   RESEARCH NOTICE
--------------------------------------------------------- */

.research-notice {
    padding: 1rem 1.3rem;
    margin: 1rem 0 1.5rem 0;
    border-radius: 14px;
    background: linear-gradient(
        90deg,
        #fff8e1,
        #fffdf5
    );
    border-left: 5px solid #ffb300;
    color: #5d4b00;
    box-shadow: 0 5px 18px rgba(90,70,0,0.07);
}

.research-notice strong {
    color: #8a6200;
}

/* ---------------------------------------------------------
   SECTION HEADINGS
--------------------------------------------------------- */

.section-title {
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
    padding-left: 0.8rem;
    border-left: 5px solid #00a8c6;
    color: #123b63;
    font-size: 1.45rem;
    font-weight: 750;
}

.section-subtitle {
    color: #607d8b;
    margin-bottom: 1rem;
}

/* ---------------------------------------------------------
   CARDS
--------------------------------------------------------- */

.bio-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(33,150,243,0.10);
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 8px 25px rgba(30,70,100,0.08);
    transition: all 0.2s ease;
}

.bio-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(30,70,100,0.13);
}

.card-icon {
    font-size: 1.8rem;
}

.card-title {
    font-size: 1.05rem;
    font-weight: 750;
    color: #123b63;
    margin-top: 0.3rem;
}

.card-text {
    color: #607d8b;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ---------------------------------------------------------
   METRICS
--------------------------------------------------------- */

div[data-testid="stMetric"] {
    background: white;
    border-radius: 15px;
    padding: 0.85rem 1rem;
    border: 1px solid rgba(0,150,136,0.10);
    box-shadow: 0 5px 18px rgba(30,70,100,0.07);
}

div[data-testid="stMetricLabel"] {
    color: #607d8b !important;
}

div[data-testid="stMetricValue"] {
    color: #123b63 !important;
    font-weight: 750;
}

/* ---------------------------------------------------------
   TABS
--------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: rgba(255,255,255,0.75);
    padding: 7px;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(30,70,100,0.07);
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 11px;
    padding: 0.65rem 0.85rem;
    font-weight: 650;
    color: #486581;
    border: none;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(
        135deg,
        #087f8c,
        #1976a8
    ) !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {
    border-radius: 11px;
    border: 1px solid rgba(0,137,123,0.20);
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 18px rgba(0,120,150,0.16);
}

/* Primary buttons */

.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #00897b,
        #1976a8
    );
    color: white;
    border: none;
}

/* ---------------------------------------------------------
   FILE UPLOADERS
--------------------------------------------------------- */

section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.85);
    border-radius: 16px;
    padding: 0.35rem;
}

section[data-testid="stFileUploader"] > div {
    border-radius: 14px;
}

/* ---------------------------------------------------------
   EXPANDERS
--------------------------------------------------------- */

.streamlit-expanderHeader {
    border-radius: 12px !important;
    font-weight: 650 !important;
}

/* ---------------------------------------------------------
   DATAFRAMES
--------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 5px 18px rgba(30,70,100,0.07);
}

/* ---------------------------------------------------------
   INFO / SUCCESS / WARNING / ERROR
--------------------------------------------------------- */

div[data-testid="stAlert"] {
    border-radius: 13px;
}

/* ---------------------------------------------------------
   IMAGE CONTAINERS
--------------------------------------------------------- */

[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.bio-footer {
    margin-top: 2rem;
    padding: 1.5rem;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(
        135deg,
        #0b2545,
        #0d5366
    );
    color: #dffaff;
}

.bio-footer-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: white;
}

.bio-footer-text {
    margin-top: 0.35rem;
    font-size: 0.82rem;
    color: #c7e9ef;
}

/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 768px) {

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .biocolorscan-hero {
        padding: 1.5rem;
        border-radius: 18px;
    }

    .hero-title {
        font-size: 2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BIOSCOLORSCAN HERO HEADER
# ============================================================

st.markdown(
    '<div class="biocolorscan-hero"><div class="hero-content">'
    '<div class="hero-title">🔬 BioColorScan</div>'
    '<div class="hero-subtitle">Integrated Biomedical Skin Image Analysis Platform</div>'
    '<div class="hero-description">A research platform for quantitative skin image analysis, '
    'CIELAB and ITA measurements, image preprocessing, experimental colour correction, '
    'ROI analysis and deep learning based lesion analysis.</div>'
    '<div class="hero-badges">'
    '<span class="hero-badge">🎨 CIELAB / ITA</span>'
    '<span class="hero-badge">🧪 Experiments A–E</span>'
    '<span class="hero-badge">🎯 ImageJ ROI</span>'
    '<span class="hero-badge">🤖 U-Net + DenseNet</span>'
    '<span class="hero-badge">📊 Quantitative Analysis</span>'
    '</div></div></div>',
    unsafe_allow_html=True
)


# ============================================================
# RESEARCH NOTICE
# ============================================================

st.markdown(
    '<div class="research-notice">'
    '<strong>⚠️ Research and educational software</strong><br>'
    'BioColorScan is intended for research and educational use. '
    'ABCDE-style measurements, scores and deep-learning predictions '
    'are experimental outputs and are not a medical diagnosis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PLATFORM OVERVIEW CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        '<div class="bio-card">'
        '<div class="card-icon">🎨</div>'
        '<div class="card-title">Colour Analysis</div>'
        '<div class="card-text">CIELAB, pixel-wise ITA and quantitative skin colour analysis.</div>'
        '</div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        '<div class="bio-card">'
        '<div class="card-icon">🧪</div>'
        '<div class="card-title">Experiments A–E</div>'
        '<div class="card-text">Illumination correction, L stretching and experimental hair removal.</div>'
        '</div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        '<div class="bio-card">'
        '<div class="card-icon">🎯</div>'
        '<div class="card-title">ROI Analysis</div>'
        '<div class="card-text">ImageJ three-region ROI analysis with pixel-wise ITA statistics.</div>'
        '</div>',
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        '<div class="bio-card">'
        '<div class="card-icon">🤖</div>'
        '<div class="card-title">Deep Learning</div>'
        '<div class="card-text">U-Net segmentation, DenseNet classification and Grad-CAM.</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

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


def read_lab_channel(uploaded_file):
    arr = tifffile.imread(uploaded_file)

    if arr.ndim != 2:
        raise ValueError(
            f"{uploaded_file.name} must be a single 2-D TIFF channel."
        )

    return arr


def lab_channel_upload_ui(prefix=""):
    c1, c2, c3 = st.columns(3)

    with c1:
        L_file = st.file_uploader(
            "L channel (.tif/.tiff)",
            type=["tif", "tiff"],
            key=f"{prefix}_L"
        )

    with c2:
        a_file = st.file_uploader(
            "a channel (.tif/.tiff)",
            type=["tif", "tiff"],
            key=f"{prefix}_a"
        )

    with c3:
        b_file = st.file_uploader(
            "b channel (.tif/.tiff)",
            type=["tif", "tiff"],
            key=f"{prefix}_b"
        )

    return L_file, a_file, b_file


def process_uploaded_lab_channels(L_file, a_file, b_file):
    if not all([L_file, a_file, b_file]):
        return None

    L = read_lab_channel(L_file)
    a = read_lab_channel(a_file)
    b = read_lab_channel(b_file)

    if not (L.shape == a.shape == b.shape):
        raise ValueError(
            f"L, a and b shapes must match. "
            f"Received {L.shape}, {a.shape}, {b.shape}."
        )

    return L, a, b


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


# ============================================================
# ROI THREE-REGION APPROACH
# ============================================================

def roi_ita_analysis(image_rgb, roi_zip_file):
    if not ROIFILE_AVAILABLE:
        raise ImportError(
            "roifile is required for ImageJ ROI ZIP processing."
        )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    L = (
        lab[:, :, 0].astype(np.float32)
        * 100.0 / 255.0
    )

    a = (
        lab[:, :, 1].astype(np.float32)
        - 128.0
    )

    b = (
        lab[:, :, 2].astype(np.float32)
        - 128.0
    )

    results = []
    roi_visuals = []

    with zipfile.ZipFile(
        roi_zip_file,
        "r"
    ) as zip_file:

        roi_names = [
            n for n in zip_file.namelist()
            if n.lower().endswith(".roi")
        ]

        for roi_number, roi_name in enumerate(
            roi_names,
            start=1
        ):
            roi_data = zip_file.read(
                roi_name
            )

            roi = ImagejRoi.frombytes(
                roi_data
            )

            coordinates = roi.coordinates()

            if coordinates is None or len(coordinates) == 0:
                continue

            mask = np.zeros(
                image_rgb.shape[:2],
                dtype=np.uint8
            )

            polygon = np.array(
                coordinates,
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [polygon],
                255
            )

            roi_pixels = mask > 0

            L_roi = L[roi_pixels]
            b_roi = b[roi_pixels]

            valid = (
                np.isfinite(L_roi)
                &
                np.isfinite(b_roi)
                &
                (b_roi != 0)
            )

            L_roi = L_roi[valid]
            b_roi = b_roi[valid]

            if len(L_roi) == 0:
                continue

            ita = np.degrees(
                np.arctan2(
                    L_roi - 50.0,
                    b_roi
                )
            )

            results.append({
                "Image": "Uploaded image",
                "ROI": f"ROI_{roi_number}",
                "ROI_Name": roi_name,
                "Valid_Pixels": len(ita),
                "Min_ITA": round(
                    float(np.min(ita)), 2
                ),
                "Max_ITA": round(
                    float(np.max(ita)), 2
                ),
                "Mean_ITA": round(
                    float(np.mean(ita)), 2
                ),
                "Median_ITA": round(
                    float(np.median(ita)), 2
                ),
                "Std_ITA": round(
                    float(np.std(ita)), 2
                )
            })

            roi_visuals.append(
                (roi_number, roi_name, mask)
            )

    return pd.DataFrame(results), roi_visuals


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    pixel_size_mm = st.number_input(
        "Pixel size (mm/pixel)",
        min_value=0.001,
        max_value=10.0,
        value=0.05,
        step=0.001,
        format="%.3f"
    )

    st.caption(
        "Default 0.05 mm/pixel follows the supplied ABCDE.py."
    )



# ============================================================
# DEEP LEARNING MODULE
# U-Net Segmentation + DenseNet-121 Classification + Grad-CAM
# Integrated from the supplied working deep-learning application.
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    DL_AVAILABLE = True
    DL_IMPORT_ERROR = ""
except Exception as _dl_import_error:
    DL_AVAILABLE = False
    DL_IMPORT_ERROR = str(_dl_import_error)

if DL_AVAILABLE:
    DL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DL_UNET_SIZE = 256
    DL_CLASSIFIER_SIZE = 224

    # Models are kept in a local models/ directory beside this application.
    DL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    DL_UNET_PATH = os.path.join(DL_MODEL_DIR, "unet_best.pth")
    DL_DENSENET_PATH = os.path.join(DL_MODEL_DIR, "densenet_best.pth")

    # Aliases retained so the supplied model code remains checkpoint-compatible.
    DEVICE = DL_DEVICE
    UNET_SIZE = DL_UNET_SIZE
    CLASSIFIER_SIZE = DL_CLASSIFIER_SIZE
    UNET_PATH = DL_UNET_PATH
    DENSENET_PATH = DL_DENSENET_PATH

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # IMPORTANT:
        # The trained checkpoint uses the name "conv"
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024, 512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(128, 64)

        # IMPORTANT:
        # The trained checkpoint uses "output"
        self.output = nn.Conv2d(
            64,
            n_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)
# ============================================================
# LOAD U-NET
# ============================================================

@st.cache_resource
def load_unet():

    if not os.path.exists(UNET_PATH):
        return None

    model = UNet(
        n_channels=3,
        n_classes=1
    )

    checkpoint = torch.load(
        UNET_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Support different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# DENSENET-121
# ============================================================

@st.cache_resource
def load_densenet():

    if not os.path.exists(DENSENET_PATH):
        return None

    model = models.densenet121(
        weights=None
    )

    # Two classes:
    # 0 = Benign
    # 1 = Malignant
    model.classifier = nn.Linear(
        model.classifier.in_features,
        2
    )

    checkpoint = torch.load(
        DENSENET_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

unet_transform = transforms.Compose([
    transforms.Resize(
        (UNET_SIZE, UNET_SIZE)
    ),
    transforms.ToTensor(),
])


classifier_transform = transforms.Compose([
    transforms.Resize(
        (CLASSIFIER_SIZE, CLASSIFIER_SIZE)
    ),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# SEGMENTATION FUNCTION
# ============================================================

def segment_lesion(model, image):

    image_tensor = unet_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

    probability = probability.squeeze().cpu().numpy()

    # Binary mask
    mask = (
        probability > 0.5
    ).astype(np.uint8)

    return probability, mask


# ============================================================
# LESION EXTRACTION
# ============================================================

def extract_lesion(image, mask):

    image_array = np.array(image)

    # Resize mask to original image dimensions
    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    mask_image = mask_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )

    mask_array = np.array(
        mask_image
    ) > 127

    # Apply mask
    lesion = (
        image_array *
        mask_array[..., None]
    )

    return Image.fromarray(
        lesion.astype(np.uint8)
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(image, mask):

    image_array = np.array(image).astype(
        np.float32
    )

    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    mask_image = mask_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )

    mask_array = np.array(
        mask_image
    ) > 127

    overlay = image_array.copy()

    # Highlight lesion boundary approximately
    overlay[mask_array] = (
        overlay[mask_array] * 0.5
        + np.array([255, 0, 0]) * 0.5
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        overlay
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_lesion(
    model,
    image
):

    tensor = classifier_transform(
        image
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    probabilities = (
        probabilities.squeeze()
        .cpu()
        .numpy()
    )

    benign_probability = (
        probabilities[0]
    )

    malignant_probability = (
        probabilities[1]
    )

    prediction = (
        "Malignant"
        if malignant_probability >= benign_probability
        else "Benign"
    )

    return (
        prediction,
        benign_probability,
        malignant_probability
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        gradients = (
            self.gradients
            .detach()
            .cpu()
            .numpy()[0]
        )

        activations = (
            self.activations
            .detach()
            .cpu()
            .numpy()[0]
        )

        weights = np.mean(
            gradients,
            axis=(1, 2)
        )

        cam = np.zeros(
            activations.shape[1:],
            dtype=np.float32
        )

        for i, weight in enumerate(weights):

            cam += (
                weight *
                activations[i]
            )

        cam = np.maximum(
            cam,
            0
        )

        if cam.max() > 0:

            cam = (
                cam / cam.max()
            )

        return cam


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image
):

    tensor = classifier_transform(
        image
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    # DenseNet final convolutional layer
    target_layer = (
        model.features.denseblock4
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    with torch.no_grad():

        output = model(tensor)

    class_index = (
        torch.argmax(
            output,
            dim=1
        ).item()
    )

    cam = gradcam.generate(
        tensor,
        class_index
    )

    return cam, class_index

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏠 Main Analysis",
    "🤖 Deep Learning",
    "🅰️ Experiment A",
    "🅱️ Experiment B",
    "©️ Experiment C",
    "🅳 Experiment D",
    "🅴 Experiment E",
    "🎯 Three-Region ROI",
    "📊 Compare Experiments"
])

# ============================================================
# DEEP LEARNING TAB UI
# ============================================================

with tabs[1]:
    st.header("🤖 Deep Learning Skin Lesion Analysis")
    st.markdown(
        "U-Net segmentation → lesion extraction → DenseNet-121 classification → Grad-CAM"
    )
    st.info(
        "Research/educational prototype only. Deep-learning predictions are not a medical diagnosis."
    )

    if not DL_AVAILABLE:
        st.error(
            "PyTorch / torchvision could not be imported. "
            f"Install the deep-learning requirements. Details: {DL_IMPORT_ERROR}"
        )
    else:
        st.caption(
            f"Device: {DL_DEVICE} | U-Net input: {DL_UNET_SIZE}×{DL_UNET_SIZE} | "
            f"DenseNet input: {DL_CLASSIFIER_SIZE}×{DL_CLASSIFIER_SIZE}"
        )

        main_upload = st.session_state.get("main_image")
        use_main = False
        if main_upload is not None:
            use_main = st.checkbox(
                "Use the image uploaded in Main Analysis",
                value=True,
                key="dl_use_main"
            )

        if use_main and main_upload is not None:
            dl_uploaded_file = main_upload
        else:
            dl_uploaded_file = st.file_uploader(
                "Upload an image for deep-learning analysis",
                type=["jpg", "jpeg", "png"],
                key="dl_image"
            )

        if dl_uploaded_file is not None:
            try:
                dl_image = Image.open(dl_uploaded_file).convert("RGB")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(dl_image, caption="Input Image", use_container_width=True)
                with c2:
                    st.write("**Image information**")
                    st.write(f"Original size: {dl_image.width} × {dl_image.height}")
                    st.write(f"Processing device: {DL_DEVICE}")

                unet = load_unet()
                densenet = load_densenet()

                if unet is None:
                    st.warning(
                        f"U-Net model not found: `{DL_UNET_PATH}`\n\n"
                        "Place `unet_best.pth` inside the `models` folder."
                    )
                else:
                    st.subheader("1️⃣ Lesion Segmentation")
                    with st.spinner("Running U-Net segmentation..."):
                        probability, mask = segment_lesion(unet, dl_image)
                        overlay = create_overlay(dl_image, mask)
                        lesion_image = extract_lesion(dl_image, mask)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(dl_image, caption="Original", use_container_width=True)
                    with c2:
                        st.image(mask * 255, caption="Predicted Lesion Mask", use_container_width=True)
                    with c3:
                        st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

                    lesion_pixels = int(np.sum(mask))
                    total_pixels = int(mask.size)
                    lesion_percentage = lesion_pixels / total_pixels * 100 if total_pixels else 0

                    st.metric("Estimated lesion area", f"{lesion_percentage:.2f}%")
                    st.image(
                        lesion_image,
                        caption="U-Net extracted lesion",
                        width=400
                    )

                if densenet is None:
                    st.warning(
                        f"DenseNet-121 model not found: `{DL_DENSENET_PATH}`\n\n"
                        "Place `densenet_best.pth` inside the `models` folder."
                    )
                else:
                    st.subheader("2️⃣ Benign / Malignant Classification")
                    classifier_image = lesion_image if unet is not None else dl_image

                    with st.spinner("Running DenseNet-121..."):
                        (
                            prediction,
                            benign_probability,
                            malignant_probability
                        ) = classify_lesion(densenet, classifier_image)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Prediction", prediction)
                    with c2:
                        st.metric("Benign", f"{benign_probability * 100:.2f}%")
                    with c3:
                        st.metric("Malignant", f"{malignant_probability * 100:.2f}%")

                    st.progress(float(malignant_probability))
                    st.caption("Malignant probability")

                    st.subheader("3️⃣ Model Explainability")
                    try:
                        with st.spinner("Generating Grad-CAM..."):
                            cam, class_index = generate_gradcam(
                                densenet, classifier_image
                            )

                        cam_image = Image.fromarray(np.uint8(cam * 255))
                        cam_image = cam_image.resize(classifier_image.size)
                        cam_array = np.array(cam_image) / 255.0

                        fig = plt.figure()
                        plt.imshow(classifier_image)
                        plt.imshow(cam_array, alpha=0.45, cmap="jet")
                        plt.axis("off")
                        st.pyplot(fig)
                        plt.close(fig)

                        st.caption(
                            "The heatmap represents regions that influenced the "
                            "DenseNet prediction. It should not be interpreted as "
                            "a clinical indication of malignancy."
                        )
                    except Exception as e:
                        st.warning(f"Grad-CAM could not be generated: {e}")

                    st.success(
                        "Deep-learning analysis completed. The extracted lesion can "
                        "also be used as the lesion-focused image for the research analyses."
                    )

            except Exception as e:
                st.error(f"Deep-learning image processing failed: {e}")



# ============================================================
# MAIN ANALYSIS TAB
# ============================================================

with tabs[0]:
    st.header("Main Integrated Analysis")

    uploaded_file = st.file_uploader(
        "Upload skin / lesion image",
        type=[
            "jpg", "jpeg", "png", "bmp",
            "tif", "tiff"
        ],
        key="main_image"
    )

    if uploaded_file is not None:
        try:
            image_rgb = load_image(
                uploaded_file
            )

            height, width = image_rgb.shape[:2]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Width", f"{width} px")
            with c2:
                st.metric("Height", f"{height} px")
            with c3:
                st.metric(
                    "Channels",
                    image_rgb.shape[2]
                )
            with c4:
                st.metric(
                    "Data Type",
                    str(image_rgb.dtype)
                )

            st.image(
                image_rgb,
                caption="Original Image",
                use_container_width=True
            )

            st.divider()
            st.subheader("Preprocessing")

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

            st.divider()
            st.subheader("ABCDE-Style Analysis")

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

            st.divider()
            st.subheader("Pixel-wise ITA")

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

        except Exception as exc:
            st.error(
                f"Could not process image: {exc}"
            )


# ============================================================
# EXPERIMENT A
# ============================================================

with tabs[2]:
    st.header("🅰️ Experiment A — Original CIELAB")

    st.markdown(
        """
        Upload the ImageJ-exported **L, a and b TIFF channels**.
        The supplied Experiment A program calculates the mean of each
        channel and then calculates ITA from the mean L* and mean b*.
        """
    )

    files = lab_channel_upload_ui("exp_a")

    try:
        channels = process_uploaded_lab_channels(*files)

        if channels is not None:
            L, a, b = channels

            L_mean, a_mean, b_mean, ita = (
                calculate_mean_lab_ita(L, a, b)
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("L*", f"{L_mean:.2f}")
            with c2:
                st.metric("a*", f"{a_mean:.2f}")
            with c3:
                st.metric("b*", f"{b_mean:.2f}")
            with c4:
                st.metric("ITA", f"{ita:.2f}°")

            st.success(
                "Experiment A calculation completed."
            )

    except Exception as exc:
        st.error(str(exc))


# ============================================================
# EXPERIMENT B
# ============================================================

with tabs[3]:
    st.header("🅱️ Experiment B — Resize + Shades-of-Gray")

    st.markdown(
        """
        **Workflow from the supplied program:**

        Original image → Resize to **224 × 224** → Shades-of-Gray
        illumination correction with **p=6**.
        """
    )

    file = st.file_uploader(
        "Upload original image",
        type=[
            "jpg", "jpeg", "png", "bmp",
            "tif", "tiff"
        ],
        key="exp_b_image"
    )

    if file is not None:
        image = load_image(file)

        resized, sog = experiment_b_pipeline(
            image
        )

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                resized,
                caption="Resize Image — 224 × 224",
                use_container_width=True
            )

        with c2:
            st.image(
                sog,
                caption="Shades-of-Gray — p=6",
                use_container_width=True
            )

        st.download_button(
            "⬇ Download Resize Image",
            image_to_jpg_bytes(resized),
            "resize_224x224.jpg",
            "image/jpeg",
            key="b_resize_download"
        )

        st.download_button(
            "⬇ Download Shades-of-Gray Image",
            image_to_jpg_bytes(sog),
            "shades_of_gray.jpg",
            "image/jpeg",
            key="b_sog_download"
        )

    st.divider()
    st.subheader("Experiment B — ITA from LAB TIFF Channels")

    files = lab_channel_upload_ui("exp_b_lab")

    try:
        channels = process_uploaded_lab_channels(*files)

        if channels is not None:
            L, a, b = channels
            vals = calculate_mean_lab_ita(L, a, b)

            c1, c2, c3, c4 = st.columns(4)

            for col, label, value in zip(
                [c1, c2, c3, c4],
                ["L*", "a*", "b*", "ITA"],
                vals
            ):
                with col:
                    st.metric(
                        label,
                        f"{value:.2f}" +
                        ("°" if label == "ITA" else "")
                    )

    except Exception as exc:
        st.error(str(exc))


# ============================================================
# EXPERIMENT C
# ============================================================

with tabs[4]:
    st.header("©️ Experiment C — Shades-of-Gray + L Stretch")

    st.markdown(
        """
        **Workflow from the supplied program:**

        Original image → Shades-of-Gray (**p=6**) → CIELAB →
        stretch the L channel → reconstruct image.
        """
    )

    file = st.file_uploader(
        "Upload original image",
        type=[
            "jpg", "jpeg", "png", "bmp",
            "tif", "tiff"
        ],
        key="exp_c_image"
    )

    if file is not None:
        image = load_image(file)

        sog, l_stretched = (
            experiment_c_pipeline(image)
        )

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                sog,
                caption="Shades-of-Gray",
                use_container_width=True
            )

        with c2:
            st.image(
                l_stretched,
                caption="L-Stretched / Reconstructed Image",
                use_container_width=True
            )

        st.download_button(
            "⬇ Download Shades-of-Gray",
            image_to_jpg_bytes(sog),
            "shadesofgray.jpg",
            "image/jpeg",
            key="c_sog_download"
        )

        st.download_button(
            "⬇ Download L-Stretched Image",
            image_to_jpg_bytes(l_stretched),
            "Lstretched.jpg",
            "image/jpeg",
            key="c_l_download"
        )

    st.divider()
    st.subheader("Experiment C — ITA from LAB TIFF Channels")

    files = lab_channel_upload_ui("exp_c_lab")

    try:
        channels = process_uploaded_lab_channels(*files)

        if channels is not None:
            L, a, b = channels
            vals = calculate_mean_lab_ita(L, a, b)

            c1, c2, c3, c4 = st.columns(4)

            for col, label, value in zip(
                [c1, c2, c3, c4],
                ["L*", "a*", "b*", "ITA"],
                vals
            ):
                with col:
                    st.metric(
                        label,
                        f"{value:.2f}" +
                        ("°" if label == "ITA" else "")
                    )

    except Exception as exc:
        st.error(str(exc))


# ============================================================
# EXPERIMENT D
# ============================================================

with tabs[5]:
    st.header("🅳 Experiment D — Advanced Hair Removal")

    st.markdown(
        """
        **Workflow from the supplied program:**

        Blackhat morphology with line kernels → multiple orientations →
        adaptive threshold → contour filtering → dilation →
        Telea inpainting.

        Parameters from the supplied program are retained:
        kernel sizes **9 and 13**, angles **0–165° in 15° steps**,
        aspect ratio threshold **1.4**, minimum area **4**, and
        maximum area fraction **0.03**.
        """
    )

    file = st.file_uploader(
        "Upload original image",
        type=[
            "jpg", "jpeg", "png", "bmp",
            "tif", "tiff"
        ],
        key="exp_d_image"
    )

    if file is not None:
        image = load_image(file)

        result, mask, white_pct = (
            experimental_hair_removal(
                image
            )
        )

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                result,
                caption="Hair-Removed Image",
                use_container_width=True
            )

        with c2:
            st.image(
                mask,
                caption="Hair Mask",
                use_container_width=True
            )

        st.metric(
            "Hair pixel %",
            f"{white_pct:.2f}%"
        )

        st.download_button(
            "⬇ Download Hair-Removed Image",
            image_to_jpg_bytes(result),
            "hairremoved.jpg",
            "image/jpeg",
            key="d_hair_download"
        )

        st.download_button(
            "⬇ Download Hair Mask",
            image_to_png_bytes(
                cv2.cvtColor(
                    mask,
                    cv2.COLOR_GRAY2RGB
                )
            ),
            "hairmask.png",
            "image/png",
            key="d_mask_download"
        )

    st.divider()
    st.subheader("Experiment D — ITA from Hair-Removed LAB TIFF")

    files = lab_channel_upload_ui("exp_d_lab")

    try:
        channels = process_uploaded_lab_channels(*files)

        if channels is not None:
            L, a, b = channels
            vals = calculate_mean_lab_ita(L, a, b)

            c1, c2, c3, c4 = st.columns(4)

            for col, label, value in zip(
                [c1, c2, c3, c4],
                ["L*", "a*", "b*", "ITA"],
                vals
            ):
                with col:
                    st.metric(
                        label,
                        f"{value:.2f}" +
                        ("°" if label == "ITA" else "")
                    )

    except Exception as exc:
        st.error(str(exc))


# ============================================================
# EXPERIMENT E
# ============================================================

with tabs[6]:
    st.header("🅴 Experiment E — Illumination Correction + Hair Removal")

    st.markdown(
        """
        **Workflow from the supplied program:**

        Original image → Shades-of-Gray illumination correction
        (**p=6**) → advanced D-style hair removal.

        This preserves the order in the supplied Experiment E program.
        """
    )

    file = st.file_uploader(
        "Upload original image",
        type=[
            "jpg", "jpeg", "png", "bmp",
            "tif", "tiff"
        ],
        key="exp_e_image"
    )

    if file is not None:
        image = load_image(file)

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        illum_bgr = shades_of_gray(
            image_bgr,
            p=6
        )

        illum_rgb = cv2.cvtColor(
            illum_bgr,
            cv2.COLOR_BGR2RGB
        )

        hair_removed, hair_mask, white_pct = (
            experimental_hair_removal(
                illum_rgb
            )
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with c2:
            st.image(
                illum_rgb,
                caption="Illumination Corrected — p=6",
                use_container_width=True
            )

        with c3:
            st.image(
                hair_removed,
                caption="Hair Removed",
                use_container_width=True
            )

        st.image(
            hair_mask,
            caption="Hair Mask",
            use_container_width=True
        )

        st.metric(
            "Hair pixel %",
            f"{white_pct:.2f}%"
        )

        st.download_button(
            "⬇ Download Illumination Corrected",
            image_to_jpg_bytes(illum_rgb),
            "illumcorrected.jpg",
            "image/jpeg",
            key="e_illum_download"
        )

        st.download_button(
            "⬇ Download Hair Removed",
            image_to_jpg_bytes(hair_removed),
            "hairremoved.jpg",
            "image/jpeg",
            key="e_hair_download"
        )

        st.download_button(
            "⬇ Download Hair Mask",
            image_to_png_bytes(
                cv2.cvtColor(
                    hair_mask,
                    cv2.COLOR_GRAY2RGB
                )
            ),
            "hairmask.png",
            "image/png",
            key="e_mask_download"
        )

    st.divider()
    st.subheader(
        "Experiment E — ITA for Illumination-Corrected / Hair-Removed LAB"
    )

    st.markdown(
        """
        The supplied `EstimateITA - E.py` calculates separate mean-LAB
        ITA results for **Illumination-Corrected** and **Hair-Removed**
        LAB channel sets.
        """
    )

    for label, key in [
        ("Illumination-Corrected", "e_illum_lab"),
        ("Hair-Removed", "e_hair_lab")
    ]:
        st.markdown(f"**{label}**")

        files = lab_channel_upload_ui(key)

        try:
            channels = process_uploaded_lab_channels(
                *files
            )

            if channels is not None:
                L, a, b = channels
                vals = calculate_mean_lab_ita(
                    L, a, b
                )

                c1, c2, c3, c4 = st.columns(4)

                for col, name, value in zip(
                    [c1, c2, c3, c4],
                    ["L*", "a*", "b*", "ITA"],
                    vals
                ):
                    with col:
                        st.metric(
                            name,
                            f"{value:.2f}" +
                            ("°" if name == "ITA" else "")
                        )

        except Exception as exc:
            st.error(str(exc))


# ============================================================
# THREE-REGION ROI
# ============================================================

with tabs[7]:
    st.header("🎯 Three-Region ImageJ ROI ITA")

    st.markdown(
        """
        Upload the original image and an **ImageJ ROI ZIP**.
        Every `.roi` inside the ZIP is processed separately.

        For each ROI the application reports:
        **Valid Pixels, Minimum ITA, Maximum ITA, Mean ITA,
        Median ITA and Standard Deviation**.

        The supplied ROI program uses pixel-wise
        `arctan2(L - 50, b)` for the ROI ITA calculation.
        """
    )

    if not ROIFILE_AVAILABLE:
        st.error(
            "The `roifile` package is not installed. "
            "Install the supplied requirements file and restart Streamlit."
        )

    roi_image_file = st.file_uploader(
        "Upload original image",
        type=[
            "jpg", "jpeg", "png", "bmp",
            "tif", "tiff"
        ],
        key="roi_image"
    )

    roi_zip_file = st.file_uploader(
        "Upload ImageJ ROI ZIP",
        type=["zip"],
        key="roi_zip"
    )

    if (
        roi_image_file is not None
        and roi_zip_file is not None
        and ROIFILE_AVAILABLE
    ):
        try:
            roi_image = load_image(
                roi_image_file
            )

            df, roi_visuals = roi_ita_analysis(
                roi_image,
                roi_zip_file
            )

            if df.empty:
                st.warning(
                    "No valid ROI results were produced."
                )
            else:
                st.success(
                    f"Processed {len(df)} ROI(s)."
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                csv_data = df.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇ Download ROI ITA CSV",
                    csv_data,
                    "ROI_ITA_Results.csv",
                    "text/csv",
                    key="roi_csv"
                )

                st.subheader(
                    "ROI Overlay"
                )

                overlay = roi_image.copy()

                for roi_number, roi_name, mask in roi_visuals:
                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    cv2.drawContours(
                        overlay,
                        contours,
                        -1,
                        (255, 0, 0),
                        2
                    )

                    ys, xs = np.where(
                        mask > 0
                    )

                    if len(xs):
                        cx = int(np.mean(xs))
                        cy = int(np.mean(ys))

                        cv2.putText(
                            overlay,
                            f"ROI {roi_number}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA
                        )

                st.image(
                    overlay,
                    caption="ImageJ ROI Overlay",
                    use_container_width=True
                )

        except Exception as exc:
            st.error(
                f"ROI processing failed: {exc}"
            )


# ============================================================
# COMPARE EXPERIMENTS
# ============================================================

with tabs[8]:
    st.header("📊 Compare Experiments A-E")

    st.markdown(
        """
        Upload the CIELAB channel triplets produced by your
        experimental pipeline. Each row represents one condition.

        This reproduces the supplied `EstimateITA` programs while
        allowing the results to be compared in one browser page.
        """
    )

    conditions = [
        ("Experiment A", "cmp_a"),
        ("Experiment B — Resize", "cmp_b_resize"),
        ("Experiment B — Shades of Gray", "cmp_b_sog"),
        ("Experiment C — L-Stretched", "cmp_c_lstretch"),
        ("Experiment C — Shades of Gray", "cmp_c_sog"),
        ("Experiment D — Hair-Removed", "cmp_d"),
        ("Experiment E — Illumination-Corrected", "cmp_e_illum"),
        ("Experiment E — Hair-Removed", "cmp_e_hair")
    ]

    comparison_rows = []

    for label, key in conditions:
        st.markdown(f"### {label}")

        files = lab_channel_upload_ui(key)

        if all(files):
            try:
                channels = process_uploaded_lab_channels(
                    *files
                )

                L, a, b = channels

                L_mean, a_mean, b_mean, ita = (
                    calculate_mean_lab_ita(
                        L, a, b
                    )
                )

                comparison_rows.append({
                    "Condition": label,
                    "L*": round(L_mean, 2),
                    "a*": round(a_mean, 2),
                    "b*": round(b_mean, 2),
                    "ITA (degrees)": round(ita, 2)
                })

            except Exception as exc:
                st.error(
                    f"{label}: {exc}"
                )

    if comparison_rows:
        st.divider()
        st.subheader(
            "Experimental ITA Comparison"
        )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.download_button(
            "⬇ Download Experiment Comparison CSV",
            comparison_df.to_csv(
                index=False
            ),
            "Experiment_ITA_Comparison.csv",
            "text/csv",
            key="comparison_csv"
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.bar(
            comparison_df["Condition"],
            comparison_df["ITA (degrees)"]
        )

        ax.set_ylabel(
            "ITA (Degrees)"
        )

        ax.set_title(
            "ITA Comparison Across Experimental Conditions"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        ax.grid(
            axis="y"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)
    else:
        st.info(
            "Upload at least one complete L/a/b channel triplet "
            "to populate the comparison table."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="bio-footer">'
    '<div class="bio-footer-title">🔬 BioColorScan</div>'
    '<div class="bio-footer-text">'
    'Integrated Biomedical Skin Image Analysis Platform<br>'
    'CIELAB • ITA • Experiments A–E • ImageJ ROI • Deep Learning<br><br>'
    'Research and Educational Use'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)