import pandas as pd
import os
import joblib

# ---------------- PATH SETUP ----------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "model")

# ---------------- LOAD MODEL & ENCODERS ----------------
model = joblib.load(os.path.join(MODEL_DIR, "cancel_model.pkl"))
vehicle_encoder = joblib.load(os.path.join(MODEL_DIR, "vehicle_encoder.pkl"))
payment_encoder = joblib.load(os.path.join(MODEL_DIR, "payment_encoder.pkl"))

# ---------------- USER INPUT ----------------
print("\n🚕 Ride Cancellation Prediction\n")

# IMPORTANT: Normalize input to match training data
vehicle_type = input(
    "Enter vehicle type (Auto / Mini / Sedan / SUV): "
).strip().upper()

payment_method = input(
    "Enter payment method (Cash / Online): "
).strip().title()

ride_distance = float(input("Enter ride distance (in km): "))
booking_hour = int(input("Enter booking hour (0–23): "))

# ---------------- INPUT DATAFRAME ----------------
user_data = pd.DataFrame([{
    "vehicle_type": vehicle_type,
    "payment_method": payment_method,
    "ride_distance": ride_distance,
    "booking_hour": booking_hour
}])

# ---------------- SAFE ENCODING ----------------
unknown_category = False

if vehicle_type not in vehicle_encoder.classes_:
    unknown_category = True

if payment_method not in payment_encoder.classes_:
    unknown_category = True

if not unknown_category:
    user_data["vehicle_type"] = vehicle_encoder.transform(
        user_data["vehicle_type"]
    )
    user_data["payment_method"] = payment_encoder.transform(
        user_data["payment_method"]
    )

# ---------------- BUSINESS + ML LOGIC ----------------
final_decision = None
reason = None

# Rule 0: Unknown category (business override)
if unknown_category:
    final_decision = "CANCELLED"
    reason = "Unknown vehicle/payment category (business override)"

# Rule 1: Late-night rides
elif booking_hour >= 23 or booking_hour <= 5:
    final_decision = "CANCELLED"
    reason = "Late-night ride with low driver availability"

# Rule 2: Very short distance + cash payment
elif ride_distance < 2 and payment_method == "Cash":
    final_decision = "CANCELLED"
    reason = "Low driver acceptance for short cash rides"

# Rule 3: Machine Learning decision
else:
    ml_prediction = model.predict(user_data)[0]
    if ml_prediction == 1:
        final_decision = "CANCELLED"
        reason = "Machine Learning prediction"
    else:
        final_decision = "NOT CANCELLED"
        reason = "Machine Learning prediction"

# ---------------- OUTPUT ----------------
print("\n🔍 Prediction Result")
print("--------------------")
print(f"Final Decision : {final_decision}")
print(f"Decision Basis : {reason}")
