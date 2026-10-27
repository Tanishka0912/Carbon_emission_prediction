# Carbon Footprint Prediction

Predicting an individual's **carbon emission** from lifestyle, transport, energy, consumption and waste habits, using Linear Regression and regularized regression (Ridge, Lasso, ElasticNet).

## Problem statement

Understanding which everyday behaviours drive carbon emissions helps individuals and organisations target the changes that matter most. This project builds regression models to predict `CarbonEmission` and identifies the strongest drivers.

## Dataset

- **File:** `Carbon_Emission_-_LMS.csv`
- **Size:** 10,000 rows x 20 columns
- **Target:** `CarbonEmission`
- **Inputs:** body type, sex, diet, shower frequency, heating source, transport mode, vehicle type, social activity, grocery bill, air-travel frequency, monthly vehicle distance, waste bag size and count, TV/PC and internet hours, new clothes per month, energy efficiency, and two multi-value columns (`Recycling`, `Cooking_With`).

## Project structure

```
.
├── Carbon_Footprint_Prediction_a3db22_fixed.ipynb   # main notebook
├── Carbon_Emission_-_LMS.csv                        # dataset (place in the same folder)
└── README.md
```

## How to run

1. Install the requirements:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy jupyter
   ```
2. Put `Carbon_Emission_-_LMS.csv` in the same folder as the notebook.
3. In the first data-loading cell, update the `pd.read_csv(...)` path to point to your copy of the file.
4. Open the notebook and choose **Kernel > Restart & Run All**.

## Methodology

**Part 1: Linear Regression**
1. Data quality checks: missing values, duplicates, data types.
2. Missing `Vehicle Type` values (only for people who do not drive) are filled with a separate category, `No Vehicle`, instead of a mean or mode.
3. Exploratory analysis: distributions, outliers, relationships with the target, correlation.
4. Feature engineering: `Recycling` and `Cooking_With` are split into one 0/1 flag per item.
5. One-hot encoding of categorical columns (first category dropped).
6. OLS summary (statsmodels), then a scikit-learn Linear Regression on a 70/30 train/test split (`random_state=57`).
7. Assumption checks: linearity, multicollinearity (VIF), homoscedasticity (residual plot, Breusch-Pagan, Durbin-Watson) and residual normality (histogram, QQ plot, skewness, kurtosis).

**Part 2: Regularized Regression**
1. Standardize features (scaler fitted on the training set only).
2. Train Ridge, Lasso and ElasticNet with starting parameters.
3. Tune `alpha` (and `l1_ratio` for ElasticNet) with 5-fold cross-validation, scoring by mean RMSE.
4. Compare the tuned models against the baseline and select a final model.

## Key results

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear Regression (baseline) | about 0.933 | about 256 |
| Ridge / Lasso / ElasticNet (tuned) | about 0.933 | about 256 |

- The baseline model is **not overfitting** (train R² 0.9338 vs test R² 0.9335).
- Ridge, Lasso and ElasticNet perform almost identically to plain Linear Regression. Their cross-validated RMSE values differ by about 0.0005, far less than the fold-to-fold spread (about 5.9), so the choice between them is effectively a tie.

## Main findings

- **Vehicle use is the biggest driver.** Monthly distance has the strongest correlation with emissions (about 0.59), and petrol vehicles emit far more than electric ones.
- **Air travel matters.** Average emissions rise steadily from "never" to "very frequently" flying.
- **Heating source matters.** Coal and wood sit above electricity.
- **Recycling is associated with lower emissions** (all four recycling flags have negative coefficients).
- **Diet, screen time, showering and social activity have small effects.**

## Data notes and limitations

- `Cook_Grill` and `Cook_Airfryer` are identical in every row, and the `Transport` dummies overlap with the `Vehicle Type` dummies. This produces infinite VIF values. Once those redundant columns are removed for the VIF check, the maximum VIF is about 2.5, so no real multicollinearity remains. Predictions are unaffected, but the coefficients of the duplicated columns should not be interpreted separately.
- The residuals show **heteroscedasticity** (error grows with predicted emission) and a U-shaped pattern: the model over-predicts the middle of the range and under-predicts both ends. The straight-line form is missing some structure.
- Residuals have heavier tails than a normal distribution, so p-values from OLS are approximate.
- Long-distance drivers appear as outliers in `Vehicle Monthly Distance Km`. They were kept because they represent real high-emission behaviour.

## Possible next steps

- Add an interaction between driving distance and vehicle type. The effect of each kilometre depends on the vehicle.
- Try tree-based models (Random Forest, Gradient Boosting), which capture non-linear effects automatically.
- A log transform of the target was considered but is not expected to help, since predictions converted back from the log scale under-predict the largest emissions.

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, statsmodels, SciPy.
