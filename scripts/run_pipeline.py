import os
import sys
import argparse
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from src.data.load_data import load_data
from src.data.process_data import process_data
from src.features.feature_engineering import feature_engineering
from src.models.train import train_model

def main(args):
    # 1. Environment Setup
    project_root = os.getcwd()
    mlruns_path = args.mlflow_uri or f"file:{os.path.join(project_root, 'mlruns')}"
    
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # 2. Start Main Execution
    with mlflow.start_run(run_name="Treatment_Prediction_Pipeline"):
        
        # --- A. Load & Process ---
        df = load_data(args.input)
        df = process_data(df)
        print(f"Data Loaded & Processed. Shape: {df.shape}")

        # --- B. SPLIT DATA (Critical Step) ---
        # We split BEFORE encoding to prevent data leakage (Target Encoding Leakage)
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
        print(f"Data Split. Train: {train_df.shape}, Test: {test_df.shape}")

        # --- C. Feature Engineering ---
        # 1. Fit & Transform Training Data (Fit=True)
        # This learns the patterns (like Country -> Number) and returns the encoders
        train_df_encoded, preprocessors = feature_engineering(train_df, fit=True)
        
        # 2. Transform Test Data (Fit=False)
        # We use the encoders learned from Train to transform Test. 
        # Crucially, this does NOT Upsample the test set.
        test_df_encoded, _ = feature_engineering(test_df, fit=False, encoders=preprocessors)
        
        print("Feature Engineering Completed.")

        # --- D. Train Model ---
        # Pass the processed dfs and the preprocessors to be saved
        train_model(
            train_df=train_df_encoded, 
            test_df=test_df_encoded, 
            target=args.target, 
            preprocessors=preprocessors
        )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Adjust default path as needed
    p.add_argument("--input", type=str, default=os.path.join("data", "raw", "survey.csv"))
    p.add_argument("--target", type=str, default="treatment")
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--experiment", type=str, default="Treatment_Prediction_Prod")
    p.add_argument("--mlflow_uri", type=str, default=None)

    args = p.parse_args()
    main(args)