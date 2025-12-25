import pickle
import os
from load_data import load_data
from preprocess import preprocess_data
from features import create_features
from eda import run_eda
from train_model import train_models
from evaluate import evaluate_model
from feature_importance import show_feature_importance
from predict import predict_new_ride
from config import CLEAN_DATA_PATH, MODEL_PATH

def main():
    # 1️⃣ Load & prepare data
    df = load_data()
    df = preprocess_data(df)
    df = create_features(df)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # 2️⃣ EDA
    run_eda(df)

    # 3️⃣ Train models
    rf_model, lr_model, rf_acc, lr_acc, X_test, y_test, feature_cols = train_models(df)

    print(f"\n🌲 Random Forest Accuracy: {rf_acc:.3f}")
    print(f"📈 Logistic Regression Accuracy: {lr_acc:.3f}")

    # 4️⃣ Evaluate Random Forest
    evaluate_model(rf_model, X_test, y_test)

    # 5️⃣ Feature importance
    show_feature_importance(rf_model, feature_cols)

    # 6️⃣ Save model
    os.makedirs("outputs/model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)

    print("\n✅ Model saved successfully")

    # 7️⃣ Predict on NEW ride
    new_ride = {
        "hour": 23,
        "Booking_Value": 120,
        "Ride_Distance": 45.0,
        "Customer_Rating": 1.5,
        "Driver_Ratings": 1.8,
        "Payment_Method_enc": 1,
        "route": "KIA → Silk Board"
    }
    result, prob, route_rate = predict_new_ride(new_ride)

    #If needed to show cancelled part (Business overide rules are used here in real world systems)
    #  if (
    # new_ride["Ride_Distance"] > 40
    # and new_ride["Customer_Rating"] < 2.0
    # and new_ride["Driver_Ratings"] < 2.0
    # ):
    #     result = "Cancelled"
    #     prob = max(prob, 0.85)  # force high probability

    # Risk mapping    
    if prob > 0.7:
        risk = "High Risk"
        action = "Avoid assignment or provide incentive"
    elif prob > 0.4:
        risk = "Medium Risk"
        action = "Assign experienced driver"
    else:
        risk = "Low Risk"
        action = "Proceed normally"

    print("\n🚦 Route-based Prediction")
    print(f"Route                    : {new_ride['route']}")
    print(f"Historical Route Risk    : {route_rate:.2f}")
    print(f"Prediction               : {result}")
    print(f"Probability              : {prob:.6f}")
    print(f"Risk Level               : {risk}")
    print(f"Suggested Action         : {action}")
    print(f"Route cancellation rate used: {route_rate:.2f}")

if __name__ == "__main__":
    main()
