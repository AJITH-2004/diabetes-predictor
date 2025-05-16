# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Step 2: Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Step 5: Save model + scaler
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)

print("✅ Model trained and saved as diabetes_model.pkl")
