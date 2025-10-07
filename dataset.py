import numpy as np
import pandas as pd

def load_feature_matrix(feature_csv_path):
    """
    Load anonymized feature matrix.
    Columns: c1-c42, r1-r76, d1-d40, label (integers, 0-3 for 4-class)
    """
    df = pd.read_csv(feature_csv_path)
    clinical = df.iloc[:, :42].values.astype(np.float32)
    radiomics = df.iloc[:, 42:42+76].values.astype(np.float32)
    deep = df.iloc[:, 42+76:42+76+40].values.astype(np.float32)
    y = df['label'].values.astype(np.int32)
    return clinical, radiomics, deep, y