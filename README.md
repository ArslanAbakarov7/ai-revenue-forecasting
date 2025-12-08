# 📦 AI Revenue Forecasting Workflow  
### **End-to-End Machine Learning Pipeline for Monthly Revenue Prediction**

This project delivers a complete machine-learning production workflow that ingests raw invoice-level JSON files, cleans and consolidates them, engineers time-series features, trains multiple forecasting models, evaluates them, selects the best approach, and produces a deployable prediction artifact.



# 🎯 **1. Purpose of the Project**

Modern subscription and e-commerce platforms depend on **reliable revenue forecasting** to support:

- Inventory planning  
- Marketing strategy  
- Customer retention forecasting  
- Executive reporting  
- Cash-flow management  

This project answers the following business question:

> **Given historical invoice-level data, can we reliably forecast next-month total revenue?**

The project demonstrates:

- Best practices in data engineering  
- Use of supervised time-series forecasting techniques  
- Model selection based on performance metrics  
- Unit testing across ingestion, features, models, and API  
- Artifact generation for reproducible ML deployment


# 📊 **2. Data Summary & Cleaning Steps**

The original dataset consists of **21 JSON files**, each containing invoice-level purchases.

Test data is available for downloading in here: https://github.com/aavail/ai-workflow-capstone/tree/master/cs-train

Each JSON includes:

- `country`  
- `invoice`  
- `customer_id`  
- `stream_id`  
- `times_viewed`  
- `year`, `month`, `day`  
- `price`  

### ✔ Cleaning Steps Performed

| Cleaning Action | Description |
|-----------------|-------------|
| Standardized column names | Case normalization + consistent snake_case |
| Constructed datetime | Combined `year + month + day` → `date` |
| Converted prices | Ensured numerical type + absolute value to fix negative entries |
| Removed invalid rows | Dropped rows missing `price` or `date` |
| Consolidated data | Combined 21 JSON files into one unified CSV |
| Normalized duplicates | Handled repeated invoices by summing revenue |

The processed data is saved as:
artifacts/all_invoices_consolidated.csv



# 📈 **3. Data Visualization (EDA)**

Below are examples of exploratory data analysis visualizations.


### 📈 **Monthly Revenue Over Time**

Run the code below to see monthly revenue over time

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ai-revenue-forecasting/artifacts/all_invoices_consolidated.csv", parse_dates=["date"])
monthly = df.groupby(df["date"].dt.to_period("M"))["price"].sum().to_timestamp()

plt.figure(figsize=(12,5))
plt.plot(monthly.index, monthly.values)
plt.title("Monthly Revenue Over Time")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.grid(True)
plt.show()


📊 Country Contribution Breakdown

country_rev = df.groupby("country")["price"].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))
country_rev.head(10).plot(kind="bar")
plt.title("Top 10 Revenue-Contributing Countries")
plt.xlabel("Country")
plt.ylabel("Total Revenue (£)")
plt.grid(True)
plt.show()


🧮 5. Feature Engineering
This project uses supervised learning with the following engineered features:

Feature	Description
lag_1, lag_3, lag_6	Revenue values from 1, 3, and 6 months prior
Rolling means	3-month and 6-month moving averages
Percent change	1-month and 3-month revenue changes
Month cyclic encodings	sin + cos transforms for seasonality
Target	Next-month revenue

The table is output to:
artifacts/monthly_features.csv

🤖 6. Model Training & Selection
Three models are trained and evaluated:

Baseline model

Prediction = previous month's revenue (lag_1)

Random Forest Regressor

XGBoost Regressor

Performance metrics stored in:

artifacts/metrics.json
A typical output:

{
  "baseline": 22681.41,
  "rf": 48389.72,
  "xg": 36026.62,
  "best_model": "baseline"
}
The best-performing model is saved to:
artifacts/final_model.joblib

🔮 7. Making Predictions
Predict overall next-month revenue:

from ai-revenue-forecasting.api.features_and_model import predict_next_month_global
predict_next_month_global()

Predict using your own uploaded JSON file:

from google.colab import files
from ai-revenue-forecasting.api.features_and_model import predict_next_month_global

🧪 8. Running All Unit Tests
Tests cover:

Data ingestion

Feature engineering

Model training

Prediction

API endpoints

Run all tests:
pytest -q ai-revenue-forecasting/tests


▶️ 9. Running the Full Pipeline in Jupyter Notebook
1. Ingest JSON files

from ai-revenue-forecasting.api.data_ingest import ingest_all_jsons
ingest_all_jsons()

2. Build monthly features

from ai-revenue-forecasting.api.features_and_model import build_monthly_features
build_monthly_features()

3. Train & select best model

from ai-revenue-forecasting.api.features_and_model import train_select_and_save
train_select_and_save()

4. Predict next-month revenue

from ai-revenue-forecasting.api.features_and_model import predict_next_month_global
predict_next_month_global()

🌐 10. Docker & API (Optional for Future Work)
The project includes a simple app.py Flask API skeleton.
Containerization (Dockerfile + Gunicorn) can be added later following standard ML deployment patterns.
