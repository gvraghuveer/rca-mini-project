from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config import TARGET_COL, RANDOM_STATE

# Feature set WITHOUT leakage
FEATURE_COLS = [
    "hour",
    "Booking_Value",
    "Ride_Distance",
    "Customer_Rating",
    "Driver_Ratings",
    "Payment_Method_enc",
    "route_cancellation_rate"
]

def train_models(df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 🌲 Random Forest
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    # 📈 Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    return rf_model, lr_model, rf_acc, lr_acc, X_test, y_test, FEATURE_COLS
