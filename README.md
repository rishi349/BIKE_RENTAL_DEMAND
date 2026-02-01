# Bike Demand Prediction using Linear and Polynomial Regression

## 📌 Project Overview

This project aims to predict **bike rental demand (`count`)** using regression-based machine learning models.  
The pipeline explores **linear regression** and **polynomial regression (with and without interactions)** combined with **Ridge regularization**, advanced **feature engineering**, and proper **evaluation techniques** to improve prediction accuracy.

The dataset is time-dependent, so special care is taken to avoid data leakage by performing a **chronological train–test split**.

---

## 📊 Dataset Description

The dataset (`train.csv`) contains historical bike rental data with the following key attributes:

- **Datetime information** (date and hour)
- **Weather-related features** (temperature, humidity, windspeed, etc.)
- **Categorical indicators** (season, weather condition, holiday, working day)
- **Target variable**: `count` (number of bike rentals)

---

## 🧠 Feature Engineering

The following feature transformations are applied:

### 1. Time-Based Features
Extracted from the `datetime` column:
- `hour`
- `weekday`
- `month`

### 2. Cyclical Encoding
To properly represent periodic behavior:
- `hour_sin`
- `hour_cos`

This ensures continuity between hours (e.g., 23 → 0).

### 3. Derived Numerical Features
- `temp_diff` = `temp - atemp`
- `humid_ws` = `humidity × windspeed`

These capture combined environmental effects.

---

## 🔧 Preprocessing

### Numerical Features
- Standardized using **StandardScaler**

### Categorical Features
- Encoded using **OneHotEncoder**
- `handle_unknown="ignore"` prevents errors during testing

### Target Transformation
- Target variable is transformed using `log(1 + count)`
- Predictions are inverted using `expm1`
- **Smearing correction** is applied to remove bias introduced by the log transformation

---

## 🏗️ Models Implemented

### 1. Linear Regression (Baseline)
- Uses scaled numerical features + one-hot encoded categorical features
- Serves as a benchmark for more complex models

### 2. Polynomial Regression (No Interactions)
- Polynomial degrees: **2, 3, and 4**
- Only individual feature powers are included
- Interaction terms are explicitly removed
- Dimensionality controlled using:
  - Feature clipping
  - Truncated SVD (for higher degrees)

### 3. Polynomial Regression (With Interactions – Degree 2)
- Includes feature × feature interaction terms
- Higher complexity model
- Requires stronger regularization

---

## 🛡️ Regularization & Validation

- **Ridge Regression (L2 regularization)** is used for polynomial models
- Optimal regularization parameter (`alpha`) selected using:
  - **5-Fold Cross-Validation**
- Helps prevent overfitting caused by high-dimensional polynomial features

---

## 📈 Evaluation Metrics

Models are evaluated on the **test set** using:
- **Mean Squared Error (MSE)** – primary metric
- **R² Score** – goodness of fit

The best model is selected based on **lowest test MSE**.

---

## 🏆 Model Comparison

The following models are compared:

- Linear Regression
- Polynomial Regression (Degree 2, No Interactions)
- Polynomial Regression (Degree 2, With Interactions)
- Polynomial Regression (Degree 3, No Interactions)
- Polynomial Regression (Degree 4, No Interactions)

A summary table is printed showing:
- Test MSE
- Test R²
- Regularization strength (`alpha`)
- Whether interactions are used

---

## ✅ Final Outcome

- The project demonstrates how **non-linear feature transformations**, **regularization**, and **careful preprocessing** can significantly improve regression performance.
- The best-performing model is automatically selected and clearly reported.
- The pipeline is robust, reproducible, and suitable for real-world regression problems involving time-based data.

---

## 🚀 How to Run

1. Place `train.csv` in the project directory
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
