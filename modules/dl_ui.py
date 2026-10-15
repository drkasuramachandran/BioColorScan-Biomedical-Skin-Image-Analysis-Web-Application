"""Streamlit UI for the combined deep-learning section.

The models only run when the user presses the button, and the results are kept
in the session, so other interactions on the page do not re-run them.
"""

import hashlib

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd

from modules import dl_models as dl
from modules import model_info as mi


# ------------------------------------------------------------------
# Cached model loaders (None if the weight file is not available)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _cnn():
    path = dl.resolve_model_path(dl.CNN_FILE)
    return dl.load_cnn_from_h5(path) if path else None


@st.cache_resource(show_spinner=False)
def _unet():
    path = dl.resolve_model_path(dl.UNET_FILE)
    return dl.load_unet(path) if path else None


@st.cache_resource(show_spinner=False)
def _densenet():
    path = dl.resolve_model_path(dl.DENSENET_FILE)
    return dl.load_densenet(path) if path else None


def _missing(name, filename):
    st.warning(
        f"{name} weights not found. Place `{filename}` in the `models/` folder "
        f"or set HF_MODEL_REPO (and HF_TOKEN) in the app secrets."
    )


# ------------------------------------------------------------------
# Run all models once
# ------------------------------------------------------------------

def _run_all(image):
    out = {"errors": {}}

    # 5-layer CNN
    try:
        model = _cnn()
        out["cnn"] = dl.predict_cnn(model, image) if model else None
    except Exception as e:
        out["cnn"] = None
        out["errors"]["cnn"] = str(e)

    # U-Net
    lesion = image
    try:
        unet = _unet()
        if unet is not None:
            _, mask = dl.segment_lesion(unet, image)
            lesion = dl.extract_lesion(image, mask)
            out["unet"] = {
                "mask": mask,
                "overlay": dl.create_overlay(image, mask),
                "lesion": lesion,
            }
        else:
            out["unet"] = None
    except Exception as e:
        out["unet"] = None
        out["errors"]["unet"] = str(e)

    # DenseNet + Grad-CAM
    try:
        net = _densenet()
        if net is not None:
            pred, p_ben, p_mal = dl.classify_lesion(net, lesion)
            out["dense"] = {"pred": pred, "benign": p_ben, "malignant": p_mal, "input": lesion}
            try:
                out["dense"]["cam"], _ = dl.generate_gradcam(net, lesion)
            except Exception as e:
                out["errors"]["gradcam"] = str(e)
        else:
            out["dense"] = None
    except Exception as e:
        out["dense"] = None
        out["errors"]["dense"] = str(e)

    return out


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def render_deep_learning(image: Image.Image):
    st.caption(
        f"Device: {dl.DEVICE} | CNN input {dl.CNN_SIZE}×{dl.CNN_SIZE} | "
        f"U-Net input {dl.UNET_SIZE}×{dl.UNET_SIZE} | "
        f"DenseNet input {dl.CLASSIFIER_SIZE}×{dl.CLASSIFIER_SIZE}"
    )

    st.markdown("**Models and training data**")
    info = pd.DataFrame(mi.MODELS).fillna("Not recorded")
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.markdown(mi.DATASET_NOTE)
    st.caption(
        f"5-layer CNN data: [Kaggle melanoma dataset]({mi.KAGGLE_URL}). "
        f"U-Net / DenseNet-121 data: [ISIC 2016 challenge]({mi.ISIC_URL}). "
        f"Citation (ISIC 2016): {mi.ISIC_CITATION}"
    )

    key = hashlib.md5(image.tobytes()).hexdigest()

    if st.button("▶ Run deep learning analysis", key="dl_run"):
        with st.spinner("Loading models and analysing the image (the first run can take a minute)..."):
            st.session_state["dl_results"] = (key, _run_all(image))

    stored = st.session_state.get("dl_results")
    if not stored or stored[0] != key:
        st.info("Press the button to run the 5-layer CNN, U-Net, DenseNet-121 and Grad-CAM on the uploaded image.")
        return

    res = stored[1]

    tab_cnn, tab_full = st.tabs([
        "5-Layer CNN (quick classifier)",
        "U-Net → DenseNet-121 → Grad-CAM",
    ])

    # ---------------- 5-layer CNN ----------------
    with tab_cnn:
        if "cnn" in res["errors"]:
            st.error(f"5-layer CNN analysis failed: {res['errors']['cnn']}")
        elif res["cnn"] is None:
            _missing("5-layer CNN", dl.CNN_FILE)
        else:
            prob, pred = res["cnn"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", pred)
            c2.metric("Malignant probability", f"{prob * 100:.2f}%")
            c3.metric("Benign probability", f"{(1 - prob) * 100:.2f}%")
            st.progress(min(max(prob, 0.0), 1.0))

    # ---------------- U-Net + DenseNet + Grad-CAM ----------------
    with tab_full:
        st.subheader("1. Lesion segmentation (U-Net)")
        if "unet" in res["errors"]:
            st.error(f"U-Net failed: {res['errors']['unet']}")
        elif res["unet"] is None:
            _missing("U-Net", dl.UNET_FILE)
        else:
            u = res["unet"]
            c1, c2, c3 = st.columns(3)
            c1.image(image, caption="Original", use_container_width=True)
            c2.image(u["mask"] * 255, caption="Predicted lesion mask", use_container_width=True)
            c3.image(u["overlay"], caption="Segmentation overlay", use_container_width=True)
            st.metric("Estimated lesion area", f"{u['mask'].mean() * 100:.2f}%")
            st.image(u["lesion"], caption="Extracted lesion", width=400)

        st.subheader("2. Benign / malignant classification (DenseNet-121)")
        if "dense" in res["errors"]:
            st.error(f"DenseNet-121 failed: {res['errors']['dense']}")
        elif res["dense"] is None:
            _missing("DenseNet-121", dl.DENSENET_FILE)
        else:
            d = res["dense"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", d["pred"])
            c2.metric("Benign", f"{d['benign'] * 100:.2f}%")
            c3.metric("Malignant", f"{d['malignant'] * 100:.2f}%")
            st.progress(min(max(d["malignant"], 0.0), 1.0))

            st.subheader("3. Explainability (Grad-CAM)")
            if "cam" in d:
                cam_img = Image.fromarray(np.uint8(d["cam"] * 255)).resize(d["input"].size)
                fig = plt.figure()
                plt.imshow(d["input"])
                plt.imshow(np.array(cam_img) / 255.0, alpha=0.45, cmap="jet")
                plt.axis("off")
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "The heatmap shows regions that influenced the DenseNet "
                    "prediction. It is not a clinical indication of malignancy."
                )
            else:
                st.warning(f"Grad-CAM could not be generated: {res['errors'].get('gradcam', 'unknown error')}")

    if res.get("cnn") and res.get("dense") and res["cnn"][1] != res["dense"]["pred"]:
        st.info(
            f"The 5-layer CNN predicted **{res['cnn'][1]}** and DenseNet-121 predicted "
            f"**{res['dense']['pred']}**. The two models were trained on different "
            "datasets and see different inputs, so they can disagree. Neither result "
            "is a diagnosis."
        )

    st.caption(
        "Deep-learning outputs are research/educational results and must not "
        "be interpreted as a clinical diagnosis."
    )
