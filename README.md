# 🚕 Ride Cancellation Analysis & Prediction

## 📌 Project Overview
Ride cancellations reduce customer satisfaction and cause operational losses for ride-hailing platforms like Ola and Uber.  
This project analyzes historical ride booking data to understand **when and why cancellations occur** and builds a **machine learning model** to predict the likelihood of cancellation for a new ride request.

Instead of using raw pickup and drop location encodings, the project leverages **route-based historical cancellation risk**, which provides a more meaningful and realistic representation of location behavior.

---

## 🎯 Objectives
- Analyze booking and cancellation patterns
- Identify peak booking hours
- Understand key factors influencing ride cancellations
- Build a predictive machine learning model
- Incorporate **route-level cancellation history**
- Translate predictions into **actionable business decisions**

---

## 📊 Dataset
- **Source**: Ola & Uber Ride Booking Dataset (Kaggle)
- **Records**: ~103,000 rides
- **Target Variable**: `is_cancelled`
  - `0` → Not Cancelled
  - `1` → Cancelled

### Final Features Used
- Booking hour
- Ride distance
- Booking value
- Customer rating
- Driver rating
- Payment method
- **Route cancellation rate**

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Removed irrelevant identifiers
- Converted date and time columns
- Extracted booking hour
- Handled missing values
- Created binary target variable `is_cancelled`

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Booking volume by hour
- Cancellation distribution
- Ride distance vs cancellation trends
- Visualizations saved to `outputs/plots/`

---

### 3️⃣ Feature Engineering

#### 🔹 Route Cancellation Rate
Instead of using raw pickup and drop location encodings, a **route-based feature** was created:

route_cancellation_rate =
(Number of cancelled rides on a route) /
(Total rides on that route)


This captures historical cancellation behavior for a given pickup–drop route and avoids misleading numerical encodings of locations.

---

### 4️⃣ Machine Learning Models
Two supervised learning models were trained and compared:

| Model | Accuracy |
|------|----------|
| Random Forest Classifier | ~90% |
| Logistic Regression | ~90% |

**Random Forest** was selected for final predictions due to its robustness and ability to model non-linear relationships.

---

### 5️⃣ Model Evaluation
- Overall accuracy ≈ **90%**
- High recall for cancelled rides (~96%)
- No data leakage
- Good generalization on unseen data

---

### 6️⃣ Feature Importance (Key Insights)

| Feature | Importance |
|-------|------------|
| Ride Distance | Highest |
| Payment Method | Very High |
| Booking Value | Moderate |
| Route Cancellation Rate | Contextual |
| Ratings & Time | Supporting |

**Insight**:  
Long-distance rides, certain payment methods, and historically risky routes are more likely to be cancelled.

---

## 🔮 Route-Based Prediction (New Ride)

The system predicts cancellation risk for a new booking using booking details and route history.

### Example Output
🚦Route-based Prediction:\
Route : KIA → Silk Board\
Historical Route Risk : 0.28\
Prediction : Not Cancelled\
Probability : 0.003421\
Risk Level : Low Risk\
Suggested Action : Proceed normally


---

## 🚦 Business Interpretation

| Risk Level | Suggested Action |
|----------|-----------------|
| Low Risk | Proceed normally |
| Medium Risk | Assign experienced driver |
| High Risk | Offer incentives or avoid assignment |

A **business-rule layer** can optionally override ML predictions for extreme cases, reflecting real-world decision systems.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Pickle (model persistence)

---

## ▶️ How to Run the Project

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python src/main.py
```
---

## 📁 Project Structure
```bash
ride-cancellation-analysis/
│
├── data/
│   ├── raw/               # Original dataset
│   │   └── Bookings.csv
│   └── processed/         # Cleaned and feature-engineered data
│       └── clean_data.csv
│
├── src/
│   ├── load_data.py       # Load dataset
│   ├── preprocess.py      # Data cleaning and preprocessing
│   ├── features.py        # Feature engineering (route risk, encoding)
│   ├── eda.py             # Exploratory data analysis and plots
│   ├── train_model.py     # Model training and comparison
│   ├── evaluate.py        # Model evaluation metrics
│   ├── predict.py         # Prediction for new ride input
│   └── main.py            # End-to-end pipeline execution
│
├── outputs/
│   ├── plots/             # Saved EDA visualizations
│   └── model/
│       └── cancellation_model.pkl
│
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Files to ignore in Git
└── .venv/                 # Virtual environment (not committed)

```
---
## 🚀 Future Enhancements

- Use geospatial distance instead of encoded locations

- Include weather and traffic conditions

- Deploy as a web application

- Improve route modeling using clustering techniques

---

## 🎓 Conclusion

- This project demonstrates how data analysis and machine learning can be used to understand and predict ride cancellations.
- By combining historical patterns, route-level behavior, and predictive modeling, the system provides realistic and actionable insights for ride-hailing platforms.


## ‼️Note:
- Trained model files are excluded from version control due to GitHub file size limits.
- Run `python src/main.py` to generate the model locally.