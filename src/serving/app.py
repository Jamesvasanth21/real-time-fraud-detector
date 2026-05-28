import os
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
from feast import FeatureStore

# Global variables to cache model and feature store connections across requests
MODEL = None
PREPROCESSOR = None
FEATURE_STORE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI Lifespan handler managing startup and shutdown cleanly."""
    global MODEL, PREPROCESSOR, FEATURE_STORE
    try:
        # 1. Initialize Feature Store connection
        FEATURE_STORE = FeatureStore(repo_path="./feature_store")
        
        experiment_id = "1"
        base_mlruns_path = f"./mlruns/{experiment_id}"
        
        if not os.path.exists(base_mlruns_path):
            raise FileNotFoundError(f"The tracking directory '{base_mlruns_path}' does not exist inside the container.")
            
        # 2. Extract the latest hex Run ID folder to locate the PREPROCESSOR
        run_ids = [
            d for d in os.listdir(base_mlruns_path) 
            if os.path.isdir(os.path.join(base_mlruns_path, d)) and len(d) == 32
        ]
        
        if not run_ids:
            raise FileNotFoundError(f"No valid MLflow run hashes found inside '{base_mlruns_path}' for the preprocessor.")
            
        run_ids.sort(key=lambda x: os.path.getmtime(os.path.join(base_mlruns_path, x)))
        latest_run_id = run_ids[-1]
        preprocessor_path = f"{base_mlruns_path}/{latest_run_id}/artifacts/preprocessor.pkl"
        
        # 3. Locate the MODEL folder inside the nested 'models' sub-directory
        models_dir = os.path.join(base_mlruns_path, "models")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Expected a 'models' directory at {models_dir} but it wasn't found.")
            
        model_runs = [
            d for d in os.listdir(models_dir) 
            if os.path.isdir(os.path.join(models_dir, d))
        ]
        
        if not model_runs:
            raise FileNotFoundError(f"No model runtime folders found inside '{models_dir}'.")
            
        model_runs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
        latest_model_folder = model_runs[-1]
        
        # MLflow saves the unified model artifacts inside this run's artifacts folder
        model_uri = f"{models_dir}/{latest_model_folder}/artifacts"
        
        print(f"✅ Latest Run ID for Preprocessor: {latest_run_id}")
        print(f"✅ Latest Model Folder Identified: {latest_model_folder}")
        print(f"📦 Attempting to load model from: {model_uri}")
        print(f"📦 Attempting to load preprocessor from: {preprocessor_path}")
        
        # 4. Load assets straight into application RAM memory wrapper
        MODEL = mlflow.xgboost.load_model(model_uri)
        with open(preprocessor_path, "rb") as f:
            PREPROCESSOR = pickle.load(f)
            
        print("🚀 Production artifacts and feature store successfully cached in application memory.")
    except Exception as e:
        import traceback
        print(f"❌ Error during server initialization: {str(e)}")
        traceback.print_exc()
        
    yield
    print("🛑 Shutting down server inference application context.")

# Initialize FastAPI application using the explicit lifespan context
app = FastAPI(title="Real-Time Fraud Detection Service", version="1.0.0", lifespan=lifespan)

class TransactionPayload(BaseModel):
    user_id: str
    transaction_id: str
    amount: float
    device_type: str

@app.get("/health")
def health_check():
    if MODEL is not None and FEATURE_STORE is not None:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Service Unhealthy/Initializing")

@app.post("/predict")
def predict_fraud(payload: TransactionPayload):
    if MODEL is None or FEATURE_STORE is None:
        raise HTTPException(status_code=503, detail="Model assets not ready.")
    
    try:
        # 1. Define entities to look up historical features from Feast Online Store
        entity_rows = [{"user_id": payload.user_id}]
        
        # 2. Query low-latency online store for user profile contexts
        online_features = FEATURE_STORE.get_online_features(
            features=[
                "user_profile_features:signup_country",
                "user_profile_features:kyc_status"
            ],
            entity_rows=entity_rows
        ).to_dict()
        
        # 3. Combine incoming real-time payload features with online historical features
        input_data = pd.DataFrame({
            "signup_country": [online_features["signup_country"][0]],
            "kyc_status": [online_features["kyc_status"][0]],
            "amount": [payload.amount],
            "device_type": [payload.device_type]
        })
        
        # Catch cases where user_id does not exist in our feature store
        if input_data["signup_country"].iloc[0] is None:
            input_data["signup_country"] = "UNKNOWN"
            input_data["kyc_status"] = "UNKNOWN"

        # 4. Process raw strings using our serialized pipeline preprocessor
        processed_features = PREPROCESSOR.transform(input_data)
        
        # 5. Run low-latency prediction inference
        prediction_prob = float(MODEL.predict_proba(processed_features)[:, 1][0])
        is_fraud = bool(prediction_prob > 0.5)
        
        return {
            "transaction_id": payload.transaction_id,
            "user_id": payload.user_id,
            "fraud_probability": round(prediction_prob, 4),
            "is_fraud": is_fraud
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")