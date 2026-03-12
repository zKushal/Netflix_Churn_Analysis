"""
Example script showing how to use the Netflix Churn Prediction Model

Usage:
    python example_usage.py
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the complete model bundle"""
    model_bundle = joblib.load('netflix_churn_model.pkl')
    return model_bundle

def predict_churn(user_data, model_bundle):
    """
    Predict churn for a user
    
    Args:
        user_data: dict with keys ['Age', 'Watch_Time_Hours', 'Country', 'Subscription_Type', 'Favorite_Genre']
        model_bundle: loaded model bundle
    
    Returns:
        dict with prediction and probability
    """
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    label_encoders = model_bundle['label_encoders']
    feature_names = model_bundle['feature_names']
    
    # Create DataFrame
    df = pd.DataFrame([user_data])
    
    # Encode categorical features
    for col in ['Country', 'Subscription_Type', 'Favorite_Genre']:
        df[col] = label_encoders[col].transform(df[col])
    
    # Ensure correct column order
    df = df[feature_names]
    
    # Scale features
    scaled_data = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    churn_probability = model.predict_proba(scaled_data)[0, 1]
    
    return {
        'prediction': 'Churned' if prediction == 1 else 'Active',
        'churn_probability': churn_probability,
        'active_probability': 1 - churn_probability
    }

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Netflix Churn Prediction Model - Example Usage")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model bundle...")
    model_bundle = load_model()
    print("   ✅ Model loaded successfully")
    
    # Example 1: Active User
    print("\n2. Predicting for Sample Users:")
    print("-" * 60)
    
    active_user = {
        'Age': 45,
        'Watch_Time_Hours': 450,
        'Country': 'US',
        'Subscription_Type': 'Premium',
        'Favorite_Genre': 'Action'
    }
    
    result1 = predict_churn(active_user, model_bundle)
    print(f"\nUser Profile: {active_user}")
    print(f"Prediction: {result1['prediction']}")
    print(f"Churn Probability: {result1['churn_probability']:.2%}")
    print(f"Active Probability: {result1['active_probability']:.2%}")
    
    # Example 2: At-Risk User
    print("\n" + "-" * 60)
    
    at_risk_user = {
        'Age': 22,
        'Watch_Time_Hours': 45,
        'Country': 'India',
        'Subscription_Type': 'Basic',
        'Favorite_Genre': 'Comedy'
    }
    
    result2 = predict_churn(at_risk_user, model_bundle)
    print(f"\nUser Profile: {at_risk_user}")
    print(f"Prediction: {result2['prediction']}")
    print(f"Churn Probability: {result2['churn_probability']:.2%}")
    print(f"Active Probability: {result2['active_probability']:.2%}")
    
    # Example 3: Batch Prediction
    print("\n" + "-" * 60)
    print("\n3. Batch Prediction (Multiple Users):")
    print("-" * 60)
    
    batch_users = [
        {
            'Age': 30,
            'Watch_Time_Hours': 200,
            'Country': 'US',
            'Subscription_Type': 'Standard',
            'Favorite_Genre': 'Drama'
        },
        {
            'Age': 50,
            'Watch_Time_Hours': 500,
            'Country': 'UK',
            'Subscription_Type': 'Premium',
            'Favorite_Genre': 'Documentary'
        },
        {
            'Age': 25,
            'Watch_Time_Hours': 20,
            'Country': 'Canada',
            'Subscription_Type': 'Basic',
            'Favorite_Genre': 'Horror'
        },
    ]
    
    results = []
    for i, user in enumerate(batch_users, 1):
        result = predict_churn(user, model_bundle)
        results.append(result)
        print(f"\nUser {i}: {result['prediction']} "
              f"(Churn: {result['churn_probability']:.2%})")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)
