"""Compare BioColorScan Experiments A-E from the shared original image.

The A-E module stores its automatically generated CIELAB/ITA summary in
Streamlit session state. This page consumes that summary; it deliberately does
not ask for separate L*, a*, or b* TIFF uploads and does not regenerate ITA maps.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

RESULTS_KEY = "biocolor_ae_results"


def render_compare_experiments():
    st.markdown(
        "Compare the mean CIELAB values and mean-channel ITA calculated in "
        "Experiments A–E above. The original image is shared, and the L*, a*, "
        "and b* channels are generated automatically within each experiment. "
        "No channel TIFF uploads are required here."
    )

    records = st.session_state.get(RESULTS_KEY, [])
    if not records:
        st.info(
            "Upload an image at the top of the page and open BioColorScan "
            "Experiments A–E above. The comparison will populate automatically."
        )
        return

    df = pd.DataFrame(records)
    required = [
        "Experiment", "Mean L*", "Mean a*", "Mean b*",
        "Mean-channel ITA (degrees)",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        st.warning("The saved experiment summary is incomplete. Rerun Experiments A–E.")
        return

    df = df[required].copy()
    numeric_columns = required[1:]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    st.subheader("A–E numerical comparison")
    st.dataframe(
        df.style.format({column: "{:.3f}" for column in numeric_columns}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download comparison CSV",
        df.to_csv(index=False).encode("utf-8"),
        "BioColorScan_Experiments_A_to_E_Comparison.csv",
        "text/csv",
        key="comparison_csv_shared_results",
    )

    valid = df[np.isfinite(df["Mean-channel ITA (degrees)"])].copy()
    if valid.empty:
        st.warning("No finite ITA values are available to plot.")
        return

    st.subheader("Mean-channel ITA across experiments")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(
        valid["Experiment"],
        valid["Mean-channel ITA (degrees)"],
        color="#25a6a6",
        edgecolor="#d7f5f5",
        linewidth=0.6,
    )
    ax.axhline(0, color="#6c858a", linewidth=0.8)
    ax.set_ylabel("ITA (degrees)")
    ax.set_xlabel("Experimental condition")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    for bar, value in zip(bars, valid["Mean-channel ITA (degrees)"]):
        ax.annotate(
            f"{value:.2f}°",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4 if value >= 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "This section shows a single summary table and ITA comparison chart. "
        "It does not create another ITA map, so map visualizations elsewhere "
        "in the application are not duplicated here."
    )
