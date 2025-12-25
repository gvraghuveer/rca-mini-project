import pickle
import pandas as pd
from config import MODEL_PATH, CLEAN_DATA_PATH

FEATURE_COLS = [
    "hour",
    "Booking_Value",
    "Ride_Distance",
    "Customer_Rating",
    "Driver_Ratings",
    "Payment_Method_enc",
    "route_cancellation_rate"
]

def predict_new_ride(input_data):
    # Work on a copy
    data = input_data.copy()

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load historical data
    df_hist = pd.read_csv(CLEAN_DATA_PATH)

    # Route lookup
    route = data["route"]
    route_data = df_hist[df_hist["route"] == route]

    if len(route_data) > 0:
        route_rate = route_data["route_cancellation_rate"].iloc[0]
    else:
        route_rate = df_hist["route_cancellation_rate"].mean()

    data["route_cancellation_rate"] = route_rate

    # Remove helper column
    data.pop("route")

    # 🔥 FILL MISSING FEATURES BEFORE MODEL INPUT
    for col in FEATURE_COLS:
        if col not in data:
            data[col] = df_hist[col].mean()

    # Create input dataframe
    df_input = pd.DataFrame([data])[FEATURE_COLS]

    # # DEBUG (keep this for now)
    # print("\nDEBUG INPUT TO MODEL:")
    # print(df_input)

    # Prediction
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    status = "Cancelled" if pred == 1 else "Not Cancelled"
    return status, prob, route_rate
