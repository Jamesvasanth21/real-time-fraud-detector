import os
import yaml
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from feast import FeatureStore

import sys

# Adds the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Ensure utils can be imported cleanly
from src.pipelines.utils import FraudFeaturePreprocessor

def run_training_pipeline():
    # 1. Load Configurations
    with open("config/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Point to our historical ground-truth target file
    # In production, this would be your analytics table containing transaction outcomes
    tx_df = pd.read_parquet("feature_store/data/transactions.parquet")
    
    # Feast needs an entity dataframe to align point-in-time features
    entity_df = tx_df[['user_id', 'event_timestamp', 'is_fraud']].copy()
    
    # 3. Retrieve Historical Features from Feast
    print("⏳ Retrieving historical features from Feast Store...")
    store = FeatureStore(repo_path="./feature_store")
    training_features = store.get_historical_features(
        entity_df=entity_df,
        features=config["features"]
    ).to_df()
    
    # 4. Train/Test Split
    X = training_features.drop(columns=['is_fraud'])
    y = training_features['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Fit Preprocessor
    preprocessor = FraudFeaturePreprocessor()
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. Initialize and Train Model within an MLflow Run Context
    mlflow.set_experiment("Real-Time-Fraud-Detection")
    
    with mlflow.start_run() as run:
        print(f"🚀 MLflow Run Started. Run ID: {run.info.run_id}")
        
        # Log training configuration profiles
        mlflow.log_params(config["model_params"])
        mlflow.log_param("num_features", len(config["features"]))
        
        # Model initialization
        model = XGBClassifier(**config["model_params"], random_state=42)
        model.fit(X_train_processed, y_train)
        
        # Evaluations
        preds = model.predict(X_test_processed)
        probs = model.predict_proba(X_test_processed)[:, 1]
        
        # Key metrics for skewed/unbalanced data sets
        auc_roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        print(f"📈 Evaluation Results -> ROC-AUC: {auc_roc:.4f} | PR-AUC (Avg Precision): {pr_auc:.4f}")
        
        mlflow.log_metric("roc_auc", auc_roc)
        mlflow.log_metric("pr_auc", pr_auc)
        
        # Save structural preprocessor components alongside model for serialization matching
        import pickle
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
        mlflow.log_artifact("preprocessor.pkl")
        os.remove("preprocessor.pkl")
        
        # Log the trained signature-bound model artifact directly
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="fraud_model",
            registered_model_name="XGBoost-Fraud-Model"
        )
        
        print("✅ Pipeline executed. Model tracked and safely registered within MLflow.")

if __name__ == "__main__":
    run_training_pipeline()