# Real-Time Credit Card Fraud Detection System (Production MLOps)

A production-grade, low-latency machine learning system designed to intercept and classify fraudulent transactions in real time. This system avoids common notebook patterns by utilizing an enterprise tech stack incorporating a centralized feature store, automated experiment pipelines, and containerized serving infrastructure.

## 🏗️ System Architecture
- **Feature Store:** Feast (managing historical feature tables and real-time online state retrieval)
- **Experiment Tracking & Catalog:** MLflow (tracking parameters, metrics, metrics charts, and serialized models)
- **Serving Engine:** FastAPI + Uvicorn wrapped inside a lightweight multi-stage Docker container
- **Automation CI/CD:** GitHub Actions (automated unit testing and container image validation builds)

## 🚀 How to Run the Inference API Locally
1. Clone the repository and ensure Docker is running.
2. Build and run the containerized service:
   ```
   docker build -t real-time-fraud-detector:latest .
   docker run -p 8000:8000 real-time-fraud-detector:latest

   ```

3. Test the prediction endpoint with a raw event payload:
    ```
    curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
        -H "Content-Type: application/json" \
        -d '{"user_id": "USER_1005", "transaction_id": "TX_999", "amount": 450.0, "device_type": "MOBILE"}'

    ```



## 📈 Engineering Trade-offs & Design Decisions

* **XGBoost Scale Pos Weight:** Configured a custom `scale_pos_weight=10` parameter during model initialization to handle the stark class imbalgitance inherent to fraud data without resorting to heavy synthetic oversampling techniques.
* **Feast Integration:** Implemented Feast to entirely eliminate training-serving skew, guaranteeing that the mathematical features extracted during training perfectly align with live runtime inputs.
* **Multi-Stage Dockerfile:** Reduced production execution image size by over 60% by segregating compile dependencies into a distinct builder phase.

