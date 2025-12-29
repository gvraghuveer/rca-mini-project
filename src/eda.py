import pandas as pd
import matplotlib.pyplot as plt
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Load raw data
df = pd.read_csv(os.path.join(DATA_DIR, "raw_rides.csv"))

# ---------------- BASIC INFO ----------------
summary = []
summary.append(f"Dataset Shape: {df.shape}")
summary.append("\nColumns:\n" + ", ".join(df.columns))
summary.append("\nMissing Values:\n" + str(df.isnull().sum()))

with open(os.path.join(PLOTS_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary))

# ---------------- CANCELLATION DISTRIBUTION ----------------
cancel_counts = df["Booking_Status"].value_counts()

plt.figure()
cancel_counts.plot(kind="bar")
plt.title("Booking Status Distribution")
plt.xlabel("Booking Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "booking_status_distribution.png"))
plt.close()

# ---------------- BOOKINGS BY HOUR ----------------
# Combine Date + Time
df["booking_datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

df["booking_hour"] = df["booking_datetime"].dt.hour

plt.figure()
df["booking_hour"].value_counts().sort_index().plot(kind="line")
plt.title("Bookings by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "bookings_by_hour.png"))
plt.close()

print("✅ EDA completed successfully. Outputs saved in data/plots/")
