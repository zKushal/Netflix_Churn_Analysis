# Netflix User Churn Analysis and Streamlit Deployment

This project analyzes Netflix user behavior, trains churn prediction models, exports a bundled model artifact, and serves predictions through a Streamlit app.

## Overview

- Dataset size: 25,000 Netflix users
- Churn rule in the notebook: users inactive for more than 90 days are labeled as churned
- Deployed artifact: `model/netflix_churn_model.pkl`
- Deployed interface: `app.py`

## Current Project Structure

```
├── .streamlit/
│   └── config.toml
├── app.py
├── data/
│   └── netflix_users.csv
├── model/
│   └── netflix_churn_model.pkl
├── noteook/
│   └── netflix.ipynb
├── README.md
└── requirements.txt
```

## Dataset Features

| Feature | Description |
|---|---|
| User_ID | Unique identifier |
| Name | User name |
| Age | User age |
| Country | Country of residence |
| Subscription_Type | Subscription plan |
| Watch_Time_Hours | Total hours watched |
| Favorite_Genre | Preferred genre |
| Last_Login | Date of last login |

## Model Pipeline

The notebook includes:

1. Data loading and exploration
2. Feature engineering for churn labels
3. Exploratory data analysis
4. Preprocessing and label encoding
5. Baseline model comparison
6. SMOTE balancing
7. XGBoost tuning
8. Model export as a single bundled `.pkl`

The bundled model file contains:

- tuned XGBoost classifier
- StandardScaler
- label encoders
- feature name order

## Features Used in Deployment

The deployed Streamlit app predicts from these five model features:

- Age
- Country
- Subscription_Type
- Watch_Time_Hours
- Favorite_Genre

## Run Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open the local URL shown in the terminal.

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy.

Because `requirements.txt` and the bundled model artifact are already in the repo, no extra deployment steps are required.

## Notes

- The notebook remains the training workflow.
- The Streamlit app is the inference workflow.
- If you retrain the model, overwrite `model/netflix_churn_model.pkl` and redeploy.
