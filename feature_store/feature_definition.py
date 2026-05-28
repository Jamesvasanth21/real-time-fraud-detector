from datetime import timedelta
from feast import Entity, ValueType

from feast import (
    Field,
    FeatureView,
    FileSource,
    Project,
)
from feast.types import Float32, Int64, String



# 1. Define the Data Sources
user_profile_source = FileSource(
    name="user_profile_source",
    path="data/user_profiles.parquet",
    timestamp_field="event_timestamp",
)

transaction_source = FileSource(
    name="transaction_source",
    path="data/transactions.parquet",
    timestamp_field="event_timestamp",
)

# 2. Define the Primary Key Entity
user = Entity(name="user_id", value_type=ValueType.STRING, description="User ID")

# 3. Create Feature Views
# Feature View for User Profiles (Static/Slow-changing features)
user_profile_fv = FeatureView(
    name="user_profile_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="signup_country", dtype=String),
        Field(name="kyc_status", dtype=String),
    ],
    online=True,
    source=user_profile_source,
)

# Feature View for Transactions (Dynamic features)
transaction_fv = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="device_type", dtype=String),
    ],
    online=True,
    source=transaction_source,
)