# Ride Cancellation Prediction (Uber & Ola)

## 📌 Project Overview

This project aims to predict whether a ride booked on platforms like **Uber or Ola** will be **cancelled or not** using Machine Learning.  
The system is trained on historical ride booking data and predicts cancellation based on ride details such as vehicle type, payment method, ride distance, and booking time.

The project demonstrates a **complete data science pipeline**, including data exploration, data cleaning, model training, and user-based prediction.

---

## 🎯 Objectives

- To analyze ride booking data and identify patterns related to ride cancellations
- To preprocess and clean raw data for machine learning
- To build a machine learning model that predicts ride cancellation
- To allow users to input ride details and get a cancellation prediction

---

## 🗂️ Project Structure

```bash
ride-cancellation-analysis/
│
├── data/
│ ├── raw_rides.csv # Original dataset
│ ├── cleaned_rides.csv # Cleaned dataset used for training
│ └── plots/ # EDA output plots and summary
│
├── model/
│ ├── cancel_model.pkl # Trained ML model
│ ├── vehicle_encoder.pkl # Encoder for vehicle type
│ └── payment_encoder.pkl # Encoder for payment method
│
├── src/
│ ├── eda.py # Exploratory Data Analysis
│ ├── clean_data.py # Data cleaning and preprocessing
│ ├── train.py # Model training and evaluation
│ └── predict.py # User input and prediction
│
├── requirements.txt
└── README.md
```

---

## 📊 Exploratory Data Analysis (EDA)

EDA is performed using `eda.py` to understand:

- Distribution of booking statuses (Cancelled vs Completed)
- Booking trends based on time of day
- Missing values and dataset structure

All plots and summaries generated during EDA are saved inside the `data/plots/` folder.

---

## 🧹 Data Cleaning & Preprocessing

The `clean_data.py` script:

- Removes duplicate records
- Handles missing values
- Combines date and time columns
- Extracts booking hour
- Converts booking status into a binary target variable (`cancelled`)
- Saves the cleaned dataset as `cleaned_rides.csv`

---

## 🤖 Model Training

The `train.py` script:

- Loads the cleaned dataset
- Encodes categorical variables
- Splits data into training and testing sets
- Trains a **Random Forest Classifier**
- Evaluates the model using accuracy
- Saves the trained model and encoders for later use

---

## 🔮 Prediction

The `predict.py` script allows users to:

- Enter ride details manually
- Automatically preprocess the input
- Predict whether the ride will be **Cancelled** or **Not Cancelled**

The prediction is purely **machine-learning based**, without additional business rules.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## ▶️ How to Run the Project

1. **Create and activate virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run EDA**

```bash
py src/eda.py
```

4. **Clean the data**

```bash
py src/clean_data.py
```

5. **Train the model**

```bash
py src/train.py
```

6. **Run prediction**

```bash
py src/predict.py
```

---

## 🎓 Conclusion

- This project demonstrates how data analysis and machine learning can be used to understand and predict ride cancellations.
- By combining historical patterns, route-level behavior, and predictive modeling, the system provides realistic and actionable insights for ride-hailing platforms.

## ‼️Note:

- Trained model files are excluded from version control due to GitHub file size limits.
