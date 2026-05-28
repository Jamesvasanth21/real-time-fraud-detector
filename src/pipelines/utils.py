import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class FraudFeaturePreprocessor:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.categorical_cols = ['signup_country', 'kyc_status', 'device_type']
        
    def fit(self, df: pd.DataFrame):
        # Filter for columns that actually exist in the dataframe
        available_cols = [c for c in self.categorical_cols if c in df.columns]
        if available_cols:
            self.encoder.fit(df[available_cols].astype(str))
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        available_cols = [c for c in self.categorical_cols if c in df.columns]
        if available_cols:
            df_out[available_cols] = self.encoder.transform(df[available_cols].astype(str))
        
        # Drop raw identifier and metadata columns not needed for pure mathematical training
        cols_to_drop = ['user_id', 'transaction_id', 'timestamp', 'event_timestamp']
        df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns], errors='ignore')
        return df_out