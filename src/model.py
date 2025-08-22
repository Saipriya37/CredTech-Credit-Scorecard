# src/model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE   # <-- make sure to install imbalanced-learn

# -------------------------------
# Load structured stock data
# -------------------------------
stock_df = pd.read_csv("data/stock_data.csv")

# Convert columns to numeric (fix for string issue)
numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
for col in numeric_cols:
    stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

# Drop missing values
stock_df = stock_df.dropna()

# -------------------------------
# Create financial features
# -------------------------------
stock_df["Return"] = stock_df["Close"].pct_change()
stock_df["Volatility"] = stock_df["Return"].rolling(window=5).std()
stock_df["MA_5"] = stock_df["Close"].rolling(window=5).mean()
stock_df["MA_10"] = stock_df["Close"].rolling(window=10).mean()
stock_df["MA_20"] = stock_df["Close"].rolling(window=20).mean()
stock_df["Volume_Change"] = stock_df["Volume"].pct_change()

# Drop rows with NaN after feature creation
stock_df = stock_df.dropna()

# -------------------------------
# Target variable (dummy credit risk)
# -------------------------------
# Rule: If stock dropped more than 2% in a day -> "High Risk" (1) else "Low Risk" (0)
stock_df["Risk"] = np.where(stock_df["Return"] < -0.02, 1, 0)

# -------------------------------
# Train-Test Split
# -------------------------------
X = stock_df[["Volatility", "MA_5", "MA_10", "MA_20", "Volume_Change"]]
y = stock_df["Risk"]

# Balance dataset with SMOTE (oversample minority class)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# -------------------------------
# Train RandomForest Model
# -------------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print(" Model Training Done with RandomForest + SMOTE!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Explainability with SHAP
# -------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("data/shap_feature_importance.png")
print("\n SHAP feature importance saved as data/shap_feature_importance.png")
