import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(num_users=1000, num_transactions=5000):
    np.random.seed(42)
    now = datetime.utcnow()
    
    # 1. Create User Profiles (Static features)
    user_ids = [f"USER_{1000 + i}" for i in range(num_users)]
    user_profiles = pd.DataFrame({
        "user_id": user_ids,
        "signup_country": np.random.choice(["US", "IN", "UK", "CA", "DE"], size=num_users),
        "kyc_status": np.random.choice(["VERIFIED", "PENDING", "FAILED"], size=num_users, p=[0.85, 0.12, 0.03]),
        "created_timestamp": [now - timedelta(days=int(np.random.randint(30, 365))) for _ in range(num_users)]
    })
    
    # 2. Create Transaction Ledger (Dynamic features)
    tx_data = []
    for i in range(num_transactions):
        uid = np.random.choice(user_ids)
        tx_time = now - timedelta(days=int(np.random.randint(0, 30)), minutes=int(np.random.randint(0, 1440)))
        amount = float(np.random.exponential(scale=50.0) + 1.0)
        
        # Inject minor fraud correlation (e.g., high amounts or unverified users)
        is_fraud = 0
        if amount > 250 and np.random.rand() > 0.4:
            is_fraud = 1
            
        tx_data.append({
            "transaction_id": f"TX_{100000 + i}",
            "user_id": uid,
            "amount": round(amount, 2),
            "device_type": np.random.choice(["MOBILE", "DESKTOP", "TABLET"], p=[0.7, 0.25, 0.05]),
            "timestamp": tx_time,
            "is_fraud": is_fraud
        })
        
    transactions = pd.DataFrame(tx_data)
    
    # Feast requires an explicit event timestamp column
    user_profiles["event_timestamp"] = pd.to_datetime(user_profiles["created_timestamp"])
    transactions["event_timestamp"] = pd.to_datetime(transactions["timestamp"])
    
    return user_profiles, transactions

if __name__ == "__main__":
    # Save the baseline datasets as target sources for Feast
    users_df, tx_df = generate_mock_data()
    users_df.to_parquet("feature_store/data/user_profiles.parquet", index=False)
    tx_df.to_parquet("feature_store/data/transactions.parquet", index=False)
    print("✅ Mock parquet datasets successfully generated inside feature_store/data/")