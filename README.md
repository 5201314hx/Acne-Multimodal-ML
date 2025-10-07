# Multimodal Machine Learning for Acne Severity Assessment and Prognosis

## Overview

This repository contains all code and example data necessary to reproduce the multimodal fusion model for acne severity assessment as described in the manuscript:

*Multimodal Machine Learning for Acne Severity Assessment and Prognosis: A Prospective Cohort Study*

**No raw facial images or identifiable patient data are included. All data are anonymized and compliant with privacy/ethics regulations.**

## Contents

- `main.py`: Main script for training, validation, and evaluation
- `model_fusion.py`: Attention-based multimodal fusion model definition
- `dataset.py`: Data loading and preprocessing utilities
- `radiomics_extractor.py`: Script template for PyRadiomics feature extraction
- `grad_cam.py`: Grad-CAM visualization utility for model interpretability
- `feature_matrix_example.csv`: Example anonymized feature matrix for testing
- `model_weights_fusion.h5`: Trained model weights (optional, upload if available)
- `requirements.txt`: Python package requirements

## Data and Privacy

- Only anonymized feature matrices are shared. No original facial images or raw scans are included.
- For full reproducibility, contact the corresponding author for access to the complete anonymized feature set.

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run model training and evaluation:
    ```bash
    python main.py
    ```

3. For Grad-CAM visualization, see `grad_cam.py`.

## Citation

If you use this code, please cite our manuscript. For questions or collaboration, contact the corresponding author.
