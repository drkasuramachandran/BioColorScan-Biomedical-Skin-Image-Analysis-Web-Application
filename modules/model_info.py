"""
Information about the deep-learning models and the data they were trained on.

Edit this file to add test metrics (set "Reported performance" to a text such as
"Accuracy 0.91 on the 1,000 evaluation images"). Anything left as None is shown
as "Not recorded".
"""

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/hasnainjaved/"
    "melanoma-skin-cancer-dataset-of-10000-images"
)
ISIC_URL = "https://challenge.isic-archive.com/data/"

MODELS = [
    {
        "Model": "5-layer CNN",
        "Task": "Benign / malignant classification",
        "Training dataset": "Melanoma Skin Cancer Dataset of 10000 Images (Kaggle)",
        "Images": "10,000 (9,600 training, 1,000 evaluation)",
        "Classes": "Benign, Malignant",
        "Model input": "Whole image, 128 × 128",
        "Reported performance": None,
    },
    {
        "Model": "U-Net",
        "Task": "Lesion segmentation",
        "Training dataset": "ISIC 2016 (ISBI 2016 challenge)",
        "Images": None,
        "Classes": "Lesion / background",
        "Model input": "Whole image, 256 × 256",
        "Reported performance": None,
    },
    {
        "Model": "DenseNet-121 (+ Grad-CAM)",
        "Task": "Benign / malignant classification",
        "Training dataset": "ISIC 2016 (ISBI 2016 challenge)",
        "Images": None,
        "Classes": "Benign, Malignant",
        "Model input": "Lesion extracted by the U-Net, 224 × 224",
        "Reported performance": None,
    },
]

ISIC_CITATION = (
    "Gutman D, Codella NCF, Celebi E, Helba B, Marchetti M, Mishra N, Halpern A. "
    "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International "
    "Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin "
    "Imaging Collaboration (ISIC). arXiv:1605.01397, 2016."
)

DATASET_NOTE = (
    "The 5-layer CNN and the U-Net / DenseNet-121 pair were trained on **different "
    "datasets** and receive **different inputs** (whole image vs. extracted lesion). "
    "Their results are independent and are not combined or averaged."
)
