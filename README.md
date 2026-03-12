# Netflix User Churn Analysis & Prediction

A machine learning project that analyzes Netflix user data to identify churn patterns and predict which users are likely to cancel their subscriptions.

## Project Overview

This project uses a dataset of **25,000 Netflix users** to build and compare classification models for churn prediction. Users inactive for more than 90 days are labeled as churned.

**Models trained:**
- Logistic Regression
- Random Forest
- Gradient Boosting

## Dataset

- **Source:** `data/netflix_users.csv`
- **Records:** 25,000 users
- **Features:** 8

| Feature | Description |
|---|---|
| User_ID | Unique identifier |
| Name | User name |
| Age | User age |
| Country | Country of residence |
| Subscription_Type | Subscription plan (Basic/Standard/Premium) |
| Watch_Time_Hours | Total hours watched |
| Favorite_Genre | Preferred content genre |
| Last_Login | Date of last login |

## Project Structure

```
├── data/
│   └── netflix_users.csv
├── noteook/
│   └── netflix.ipynb
└── README.md
```

## Notebook Workflow

1. **Import Libraries** — pandas, numpy, sklearn, matplotlib, seaborn
2. **Load & Explore Dataset** — shape, dtypes, missing values, summary statistics
3. **Feature Engineering** — create `Days_Since_Login` and binary `Churned` label (>90 days = churned)
4. **EDA** — churn distribution, age/watch time analysis, churn rates by subscription type, country, genre, correlation heatmap
5. **Data Preprocessing** — drop non-predictive columns, label encoding, train-test split (80/20), feature scaling
6. **Model Training** — Logistic Regression, Random Forest, Gradient Boosting with cross-validation
7. **Model Evaluation** — classification reports, confusion matrices, ROC-AUC curves, metric comparison
8. **Feature Importance** — Random Forest/Gradient Boosting importances, Logistic Regression coefficients
9. **Best Model Selection** — selected based on F1 Score

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/zKushal/Netflix_Churn_Analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open and run `noteook/netflix.ipynb`
