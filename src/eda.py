import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(df):
    os.makedirs("outputs/plots", exist_ok=True)

    sns.countplot(x="Booking_Status", data=df)
    plt.title("Booking Status Distribution")
    plt.savefig("outputs/plots/booking_status.png")
    plt.clf()

    sns.countplot(x="hour", data=df)
    plt.title("Bookings by Hour")
    plt.savefig("outputs/plots/bookings_by_hour.png")
    plt.clf()

    sns.boxplot(x="Booking_Status", y="Ride_Distance", data=df)
    plt.title("Ride Distance vs Booking Status")
    plt.savefig("outputs/plots/distance_vs_status.png")
    plt.clf()
