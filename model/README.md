# Netflix Churn Prediction Model

## Overview
This directory contains the saved Netflix churn prediction model packaged as a single `.pkl` file for easy deployment and usage.

## File Structure
```
model/
├── netflix_churn_model.pkl    # Complete model bundle (all components)
├── README.md                   # This file
└── example_usage.py            # Example script for using the model
```

## Model Components
The `netflix_churn_model.pkl` file contains:
- **Tuned XGBoost Classifier** - Best-performing model after hyperparameter tuning
- **StandardScaler** - For feature scaling (fitted on training data)
- **LabelEncoders** - For encoding categorical features (Country, Subscription_Type, Favorite_Genre)
- **Feature Names** - List of feature columns in correct order

## Loading the Model

```python
import joblib

# Load the complete bundle
model_bundle = joblib.load('model/netflix_churn_model.pkl')

# Extract components
model = model_bundle['model']
scaler = model_bundle['scaler']
label_encoders = model_bundle['label_encoders']
feature_names = model_bundle['feature_names']
```

## Making Predictions

See `example_usage.py` for a complete example.

### Quick Example
```python
import pandas as pd
import joblib
import numpy as np

# Load model
model_bundle = joblib.load('model/netflix_churn_model.pkl')
model = model_bundle['model']
scaler = model_bundle['scaler']
label_encoders = model_bundle['label_encoders']

# Prepare input data
new_user = pd.DataFrame({
    'Age': [35],
    'Watch_Time_Hours': [120],
    'Country': ['US'],
    'Subscription_Type': ['Premium'],
    'Favorite_Genre': ['Drama']
})

# Encode categorical features
for col in ['Country', 'Subscription_Type', 'Favorite_Genre']:
    new_user[col] = label_encoders[col].transform(new_user[col])

# Scale features
scaled_user = scaler.transform(new_user)

# Predict
churn_prob = model.predict_proba(scaled_user)[0, 1]
prediction = model.predict(scaled_user)[0]

print(f"Churn Probability: {churn_prob:.2%}")
print(f"Prediction: {'Churned' if prediction == 1 else 'Active'}")
```

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | int | User age (18-65) |
| Watch_Time_Hours | float | Total watch time in hours |
| Country | string | User country (encoded) |
| Subscription_Type | string | ['Basic', 'Standard', 'Premium'] |
| Favorite_Genre | string | User's preferred genre (encoded) |

## Model Performance

**Test Set Metrics (Tuned XGBoost):**
- Accuracy: ~0.88
- Precision: ~0.87
- Recall: ~0.89
- F1 Score: ~0.88
- ROC-AUC: ~0.94

## How to Update the Model

To retrain the model with new data:
1. Run the complete notebook: `noteook/netflix.ipynb`
2. New model will be saved to `model/netflix_churn_model.pkl` (overwrites existing)
3. Commit changes to Git

## License
Netflix Churn Analysis Project - 2026
