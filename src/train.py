import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load cleaned data
df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_rides.csv"))

# Encode categorical columns
le_vehicle = LabelEncoder()
le_payment = LabelEncoder()

df["vehicle_type"] = le_vehicle.fit_transform(df["vehicle_type"])
df["payment_method"] = le_payment.fit_transform(df["payment_method"])

X = df.drop("cancelled", axis=1)
y = df["cancelled"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, pred))

# Save artifacts
joblib.dump(model, os.path.join(MODEL_DIR, "cancel_model.pkl"))
joblib.dump(le_vehicle, os.path.join(MODEL_DIR, "vehicle_encoder.pkl"))
joblib.dump(le_payment, os.path.join(MODEL_DIR, "payment_encoder.pkl"))

print("✅ Files saved in:", MODEL_DIR)
