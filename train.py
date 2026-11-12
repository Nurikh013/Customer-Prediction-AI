
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt



def main():
    data_path = Path(__file__).parent / "data" / "bank-full.csv"

    df = pd.read_csv(data_path, sep=";")

    print("First 5 customer records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())


    print("\nSubscription outcomes:")
    print(df["y"].value_counts())

    print("\nSubscription percentages:")
    print((df["y"].value_counts(normalize=True) * 100).round(2))

    print("\nCustomer age statistics:")
    print(df["age"].describe())

    print("\nMissing values by column:")
    print(df.isna().sum())

    print("\nUnknown values by column:")
    print((df == "unknown").sum())

    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())


    
    subscription_counts = df["y"].value_counts()

    
    plt.figure(figsize=(8, 5))

    plt.bar(
        subscription_counts.index,
        subscription_counts.values
    )

    plt.title("Bank Customer Subscription Outcomes")
    plt.xlabel("Subscription Status")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()

    # Convert subscription outcomes into numbers
    df["subscribed"] = (df["y"] == "yes").astype(int)

    # Calculate subscription rate for each job
    job_rates = (
        df.groupby("job")["subscribed"]
        .mean()
        .sort_values(ascending=False) * 100
    )

    print("\nSubscription rate by job (%):")
    print(job_rates.round(2))

    # Create a second chart
    plt.figure(figsize=(11, 5))

    plt.bar(job_rates.index, job_rates.values)

    plt.title("Subscription Rate by Job")
    plt.xlabel("Customer Job")
    plt.ylabel("Subscription Rate (%)")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


    # Create customer age groups
    age_bins = [17, 25, 35, 45, 55, 65, 100]

    age_labels = [
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "66-100"
    ]

    df["age_group"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=age_labels
    )

    # Analyze subscription outcomes by age group
    age_summary = df.groupby(
        "age_group", observed=True
    )["subscribed"].agg(
        customers="count",
        subscription_rate="mean"
    )

    age_summary["subscription_rate"] = (
        age_summary["subscription_rate"] * 100
    ).round(2)

    print("\nSubscription rates by age group:")
    print(age_summary)





if __name__ == "__main__":
    main()
