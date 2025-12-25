import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    # Encode categorical columns
    cat_cols = [
        "Payment_Method"
    ]

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    # 🔹 Create Route column
    df["route"] = df["Pickup_Location"] + " → " + df["Drop_Location"]

    # 🔹 Route cancellation rate
    route_stats = (
        df.groupby("route")["is_cancelled"]
        .mean()
        .reset_index()
        .rename(columns={"is_cancelled": "route_cancellation_rate"})
    )

    # Merge back to main dataframe
    df = df.merge(route_stats, on="route", how="left")

    return df
