import pandas as pd
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load raw data
df = pd.read_csv(os.path.join(DATA_DIR, "raw_rides.csv"))

# ---------------- CLEANING ----------------

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop rows with missing important values
df.dropna(subset=[
    "Date",
    "Time",
    "Vehicle_Type",
    "Payment_Method",
    "Ride_Distance",
    "Booking_Status"
], inplace=True)

# ---------------- FEATURE ENGINEERING ----------------

# Combine Date + Time
df["booking_datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

# Extract booking hour
df["booking_hour"] = df["booking_datetime"].dt.hour

# Create target column (1 = Cancelled, 0 = Not Cancelled)
df["cancelled"] = df["Booking_Status"].apply(
    lambda x: 1 if str(x).lower() == "cancelled" else 0
)

# ---------------- SELECT FINAL COLUMNS ----------------
df = df[[
    "Vehicle_Type",
    "Payment_Method",
    "Ride_Distance",
    "booking_hour",
    "cancelled"
]]

# Rename columns for ML friendliness
df.rename(columns={
    "Vehicle_Type": "vehicle_type",
    "Payment_Method": "payment_method",
    "Ride_Distance": "ride_distance"
}, inplace=True)

# Drop rows where booking_hour could not be extracted
df.dropna(subset=["booking_hour"], inplace=True)

# Save cleaned data
df.to_csv(os.path.join(DATA_DIR, "cleaned_rides.csv"), index=False)

print("✅ Cleaned data saved to data/cleaned_rides.csv")
