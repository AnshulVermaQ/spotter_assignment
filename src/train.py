import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_PATH = "data/train-test.csv"
VALIDATION_PATH = "data/validation.csv"
DECEMBER_PATH = "data/december-chart-inputs.csv"
PREDICTIONS_PATH = "validation_predictions.csv"
RESULTS_PATH = "validation_results.csv"


NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]


DECEMBER_NUMERIC = ["distance", "weight", "year", "month", "day_of_week"]
DECEMBER_FEATURES = DECEMBER_NUMERIC + CATEGORICAL_FEATURES


VALIDATION_MONTHS = [
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"),
]



def prepare_data(df):
    df = df.copy()

    df.loc[df["weight"] < 0, "weight"] = pd.NA

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    return df



def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])



def build_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )



def fit_and_predict(train_part, test_part, numeric_features, categorical_features):
    features = numeric_features + categorical_features
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train = preprocessor.fit_transform(train_part[features]).toarray()
    X_test = preprocessor.transform(test_part[features]).toarray()

    model = build_model()
    model.fit(X_train, np.log1p(train_part["posted_rate"]))
    predictions = np.expm1(model.predict(X_test))
    return np.maximum(predictions, 0.01)



def regression_metrics(actual, predicted):
    errors = predicted - actual
    pct_errors = np.abs(errors) / actual
    mse = np.mean(errors ** 2)
    return {
        "MAE": np.mean(np.abs(errors)),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE_%": 100 * np.mean(pct_errors),
        "MedianAPE_%": 100 * np.median(pct_errors),
    }



def run_validation(train_df):
    print("=== TIME-BASED VALIDATION ===")
    rows = []
    for start, end in VALIDATION_MONTHS:
        fold_train = train_df[train_df["date"] < start]
        fold_test = train_df[(train_df["date"] >= start) & (train_df["date"] <= end)]

        predicted = fit_and_predict(
            fold_train, fold_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES
        )
        metrics = regression_metrics(fold_test["posted_rate"].to_numpy(), predicted)
        metrics = {"test_month": start[:7], "train_rows": len(fold_train),
                   "test_rows": len(fold_test), **metrics}
        rows.append(metrics)
        print(f"{start[:7]}: train={len(fold_train):>6}  test={len(fold_test):>5}  "
              f"MAE={metrics['MAE']:.1f}  RMSE={metrics['RMSE']:.1f}  "
              f"MedianAPE={metrics['MedianAPE_%']:.2f}%")

    results = pd.DataFrame(rows)
    average = results.drop(columns="test_month").mean(numeric_only=True)
    average["test_month"] = "average"
    results = pd.concat([results, average.to_frame().T], ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("\nAverage over folds:")
    print(results.iloc[-1][["MAE", "MSE", "RMSE", "MAPE_%", "MedianAPE_%"]].astype(float).round(2).to_string())
    print(f"Saved: {RESULTS_PATH}\n")



def main():
    train_df = prepare_data(pd.read_csv(TRAIN_PATH))
    print("Development rows:", len(train_df))
    print("Date range:", train_df["date"].min().date(), "to", train_df["date"].max().date())

    run_validation(train_df)

    validation_df = pd.read_csv(VALIDATION_PATH)
    validation_prepared = prepare_data(validation_df)

    print("Training final model...")
    final_predictions = fit_and_predict(
        train_df, validation_prepared, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": final_predictions,
    }).to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved {len(final_predictions):,} predictions to {PREDICTIONS_PATH}")

    december_df = pd.read_csv(DECEMBER_PATH)
    december_prepared = prepare_data(december_df)

    december_predictions = fit_and_predict(
        train_df, december_prepared, DECEMBER_NUMERIC, CATEGORICAL_FEATURES
    )
    december_df["predicted_rate"] = december_predictions
    december_df.to_csv(DECEMBER_PATH, index=False)
    print(f"Saved {len(december_df)} December predictions to {DECEMBER_PATH}")



if __name__ == "__main__":
    main()