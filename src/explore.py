import pandas as pd

df = pd.read_csv("data/train-test.csv")
print("\n=== DATA QUALITY CHECKS ===")

print("\nNegative weights:")
print((df["weight"] < 0).sum())

print("\nNegative distances:")
print((df["distance"] < 0).sum())

print("\nZero weights:")
print((df["weight"] == 0).sum())

print("\nZero distances:")
print((df["distance"] == 0).sum())

print("\nNegative posted rates:")
print((df["posted_rate"] < 0).sum())

print("\nNegative market index:")
print((df["market_index"] < 0).sum())

print("\nNegative quote signals:")
print((df["quote_signal"] < 0).sum())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nNegative weight examples:")
print(df[df["weight"] < 0][
    ["load_id", "distance", "weight", "equipment", "posted_rate"]
].head(10))



print("\n=== WEIGHT ANALYSIS ===")

print("Positive weight statistics:")
print(df[df["weight"] > 0]["weight"].describe())

print("\nNegative weight statistics:")
print(df[df["weight"] < 0]["weight"].describe())

print("\nMissing weights:")
print(df["weight"].isna().sum())

print("\n=== POSTED RATE ANALYSIS ===")

print(df["posted_rate"].describe())

print("\nHighest posted rates:")
print(
    df.nlargest(10, "posted_rate")[
        ["load_id", "distance", "weight", "equipment", "posted_rate"]
    ]
)