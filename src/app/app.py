import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import joblib
import os
import sys

# 1. Setup Paths Correctly
# Get the absolute path to the project root (2 levels up from src/app/app.py)
# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (Telco_project)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Add project root to system path
sys.path.append(project_root)

app = FastAPI(title="Mental Health Treatment Predictor")

# --- CONFIGURATION ---
RUN_ID = "19694939bcab474da56550cf256fd252"  # Your run ID

# This points to the mlruns folder relative to the project root
mlruns_path = os.path.join(project_root, "mlruns")
MLFLOW_TRACKING_URI = f"file:{mlruns_path}"

print(f"Tracking URI set to: {MLFLOW_TRACKING_URI}")

# --- GLOBAL VARIABLES ---
model = None
preprocessors = None

class PatientData(BaseModel):
    # Inputs matching your dataset columns
    Age: int
    Gender: str
    Country: str
    self_employed: str
    family_history: str
    work_interfere: str
    no_employees: int
    remote_work: str
    tech_company: str
    benefits: str
    care_options: str
    wellness_program: str
    seek_help: str
    anonymity: str
    leave: str
    mental_health_consequence: str
    phys_health_consequence: str
    coworkers: str
    supervisor: str
    mental_health_interview: str
    phys_health_interview: str
    mental_vs_physical: str
    obs_consequence: str

@app.on_event("startup")
def load_artifacts():
    global model, preprocessors
    
    print("Loading model and artifacts from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # 1. Download the Preprocessor Artifact
        client = mlflow.tracking.MlflowClient()
        # We download to a temp folder or current dir
        local_path = client.download_artifacts(RUN_ID, "pipeline_assets/preprocessor.pkl", dst_path=".")
        
        # 2. Load the Preprocessor
        preprocessors = joblib.load(local_path)
        print("✅ Preprocessors loaded.")

        # 3. Load the Model
        model_uri = f"runs:/{RUN_ID}/model"
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded.")
        
    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}")
        # Print the path we tried so we can debug if it fails again
        print(f"Attempted to look in: {MLFLOW_TRACKING_URI}")
        raise e

@app.post("/predict")
def predict(data: PatientData):
    if not model or not preprocessors:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Convert Input to DataFrame
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        
        # 2. Apply Preprocessing
        cat_cols = preprocessors['cat_cols']
        ordinal_encoder = preprocessors['ordinal_encoder']
        
        # Filter columns that exist in input
        valid_cat_cols = [c for c in cat_cols if c in df.columns]
        df[valid_cat_cols] = ordinal_encoder.transform(df[valid_cat_cols])

        if 'Country' in df.columns and preprocessors['target_encoder']:
            target_encoder = preprocessors['target_encoder']
            df['Country'] = target_encoder.transform(df['Country'])

        # 3. Predict
        prediction = model.predict(df)
        probability = model.predict_proba(df).max()
        
        return {
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "confidence": float(probability),
            "run_id": RUN_ID
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)