import pandas as pd

def preprocess_data(df):
    df = df.drop(columns=["Booking_ID", "Customer_ID", "Vehicle Images", "Unnamed: 20"], errors="ignore")

    # Converting Date & Time
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")

    # Extracting hour
    df["hour"] = df["Time"].dt.hour

    # Main Target variable
    df["is_cancelled"] = df["Booking_Status"].apply(
        lambda x: 1 if "cancel" in x.lower() else 0
    )

    # Filling missing numeric values
    numeric_cols = [
        "Booking_Value", "Ride_Distance",
        "Customer_Rating", "Driver_Ratings",
        "V_TAT", "C_TAT"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing
    df = df.fillna("Unknown")

    return df
