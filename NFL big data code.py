import pandas as pd
import numpy as np

# pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/plays.csv", assume_missing=True)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("nathanawright24/bigdatabowl25dataset")

print("Path to dataset files:", path)