import os 
import pandas as pd

def load_data (file_path : str):

    if not os.path.exists(file_path):
        return FileNotFoundError(f" file not found at path : {file_path}")
    else :
        return pd.read_csv(file_path)
