# Skin Cancer Imaging Research Platform

Merged application combining **BioColorScan** and **SkinLesionResearchPlatform**:
image preprocessing, hair removal, CIELAB / ITA analysis, ABCDE features,
ImageJ ROI analysis, Experiments A–E, and deep learning
(5-layer CNN, U-Net segmentation, DenseNet-121 classification, Grad-CAM).

Live site: https://skincancerimaging.drkasuramachandran.com

> Research and educational use only. Not a medical diagnosis.

## Training data

| Model | Dataset |
|---|---|
| 5-layer CNN | [Melanoma Skin Cancer Dataset of 10000 Images (Kaggle)](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images): 9,600 training / 1,000 evaluation images, benign and malignant |
| U-Net, DenseNet-121, Grad-CAM | [ISIC 2016 challenge data](https://challenge.isic-archive.com/data/) |

The CNN and the U-Net / DenseNet pair were trained on different datasets, so their
outputs are independent and are not combined.

Gutman D, Codella NCF, Celebi E, Helba B, Marchetti M, Mishra N, Halpern A.
*Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International
Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin
Imaging Collaboration (ISIC).* arXiv:1605.01397, 2016.

## Model weights

Weights are NOT stored in this repository. Keep them in a private Hugging Face
model repo and set these in the Streamlit app secrets:

```
HF_MODEL_REPO = "Kasuramachandran/skin-imaging-models"
HF_TOKEN = "hf_xxx"      # read access token
```

For local use, copy `my_5layer_cnn.h5`, `unet_best.pth` and `densenet_best.pth`
into `models/` instead. Helper scripts are in `tools/`.

## Run locally

Double-click `run_app.bat` (Windows), or:

```
pip install -r requirements.txt
streamlit run app.py
```
