import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder

def feature_engineering(df: pd.DataFrame, encoders=None, fit=True):
    """
    Applies encoding and feature engineering.
    
    Args:
        df: Input dataframe.
        encoders: Dictionary of fitted encoders (required if fit=False).
        fit: Boolean. If True, learns the encodings and performs upsampling.
             If False, applies existing encodings (for test/production).
    
    Returns:
        df: Transformed dataframe.
        encoders: Dictionary of fitted encoders (to save for production).
    """
    df = df.copy()
    
    # 1. Identify Columns
    # 'treatment' is the target, so we exclude it from input features
    exclude_cols = ['Country', 'Age', 'no_employees', 'treatment']
    cat_cols = df.columns.drop(exclude_cols, errors='ignore')

    # 2. Ordinal Encoding
    if fit:
        # Initialize and Fit
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = ordinal_encoder.fit_transform(df[cat_cols])
    else:
        # Transform using existing encoder
        ordinal_encoder = encoders['ordinal_encoder']
        df[cat_cols] = ordinal_encoder.transform(df[cat_cols])
        
    print(f"Ordinal encoding {'fitted' if fit else 'applied'}")

    # 3. Target Encoding (Requires Target 'treatment')
    # Note: We only target encode 'Country' if it exists
    if 'Country' in df.columns:
        if fit:
            target_encoder = TargetEncoder()
            # We fit on Country using the 'treatment' target
            df['Country'] = target_encoder.fit_transform(df['Country'], df['treatment'])
        else:
            target_encoder = encoders['target_encoder']
            # For test/prod, we just transform (no target needed)
            df['Country'] = target_encoder.transform(df['Country'])
            
        print(f"Target encoding {'fitted' if fit else 'applied'}")
    else:
        target_encoder = None

    # 4. Upsampling (ONLY on Training Data)
    if fit:
        # Only upsample if we are fitting (Training phase)
        # This prevents 'duplicate row' leakage in the test set
        synthetic_df = df.sample(n=2000, replace=True, random_state=42)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"Upsampling done. New shape: {df.shape}")

    # 5. Return Logic
    if fit:
        # Bundle encoders to save them later
        new_encoders = {
            'ordinal_encoder': ordinal_encoder,
            'target_encoder': target_encoder,
            'cat_cols': cat_cols
        }
        return df, new_encoders
    else:
        return df, None