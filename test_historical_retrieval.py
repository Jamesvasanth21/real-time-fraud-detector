from datetime import datetime
import pandas as pd
from feast import FeatureStore

# Initialize the store pointing to our feature_store directory
store = FeatureStore(repo_path="./feature_store")

# Define the entity dataframe (the target instances we want features for)
entity_df = pd.DataFrame(
    {
        "user_id": ["USER_1001", "USER_1002"],
        "event_timestamp": [datetime.utcnow(), datetime.utcnow()],
    }
)

# Specify the features we want to pull down
features_to_fetch = [
    "user_profile_features:signup_country",
    "user_profile_features:kyc_status",
    "transaction_features:amount",
]

# Fetch historical features
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=features_to_fetch
).to_df()

print("📊 Retrieved Feature Matrix Layout:")
print(training_data.head())