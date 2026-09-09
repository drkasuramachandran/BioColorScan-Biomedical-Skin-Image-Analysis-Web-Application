# BioColorScan: Biomedical Skin Image Analysis Web Application

BioColorScan is a research oriented web application developed for biomedical skin image analysis. The application includes image preprocessing, illumination correction, hair removal, CIELAB colour analysis, pixel wise and Region of Interest based ITA analysis, lesion measurements, experimental comparison, and deep learning based lesion segmentation and classification.

The deep learning module uses a trained U Net model for lesion segmentation and DenseNet 121 for benign and malignant classification. Grad CAM is also included to visualise the regions associated with the classification result.

The application is developed using Python and Streamlit and is intended for research and educational use.
