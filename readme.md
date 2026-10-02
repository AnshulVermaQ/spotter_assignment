# Freight Rate Prediction Challenge

Predicts the posted rate (in dollars) for freight loads. The model is trained on
`data/train-test.csv` (Jan-Oct 2025) and used to predict `data/validation.csv`
(Nov-Dec 2025).

## Approach in short

- **Model:** scikit-learn `HistGradientBoostingRegressor` with absolute-error loss, predicting `log(1 + rate)`.
- **Cleaning:** negative weights are treated as missing; missing numeric values are filled with the median.
- **Validation:** time-based rolling split. For each of July, August, September and October, the model is trained on earlier months only and tested on that month.
- **Report:** `report.pdf` has the split approach, results and the December chart.

## How to run

Requires Python 3.9+. Run every command from the repository root.

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. (Optional) data quality checks and exploration
python src/explore.py

# 3. Validate, train the final model, and write predictions
python src/train.py

# 4. Check the output files and create the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv

# 5. (Optional) rebuild the PDF report
python src/make_report.py
```

## What each step produces

| Step | Output |
|---|---|
| `src/train.py` | `validation_results.csv` (rolling validation metrics), `validation_predictions.csv` (12,000 predictions), and `predicted_rate` filled in `data/december-chart-inputs.csv` |
| `score.py` | `scorer_results/candidate_december.png` |
| `src/make_report.py` | `report.pdf` |

## Files

- `src/explore.py`: data quality checks (negative/zero values, missing values, date range, extreme rates)
- `src/train.py`: cleaning, rolling validation, final training, predictions
- `src/make_report.py`: builds the PDF report from `validation_results.csv`
- `score.py`: scorer provided with the assignment (not modified)
- `data/`: input data provided with the assignment

## Validation results (rolling time split, average of 4 folds)

| MAE | RMSE | Median error | Mean error |
|---|---|---|---|
| $115.4 | $629.0 | 2.09% | 4.93% |

RMSE is much higher than MAE because about 0.7% of loads have extreme rates
that the features cannot explain; see `report.pdf` for details.
