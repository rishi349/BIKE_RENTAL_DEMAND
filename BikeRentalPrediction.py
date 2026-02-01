import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

RND = 42
np.random.seed(RND)

df = pd.read_csv("train.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

def add_time_features(df_):
    df = df_.copy()
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    return df

def add_cyclical(df_):
    df = df_.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def add_extra_features(df_):
    df = df_.copy()
    df["temp_diff"] = df["temp"] - df["atemp"]
    df["humid_ws"] = df["humidity"] * df["windspeed"]
    return df

train_df = add_time_features(train_df)
test_df  = add_time_features(test_df)
train_df = add_cyclical(train_df)
test_df  = add_cyclical(test_df)
train_df = add_extra_features(train_df)
test_df  = add_extra_features(test_df)

numeric_features = [
    "temp","atemp","humidity","windspeed",
    "hour_sin","hour_cos","temp_diff","humid_ws",
    "month","weekday"
]
categorical_features = ["season","weather","holiday","workingday"]
target = "count"

y_train = train_df[target].values
y_test  = test_df[target].values
y_train_log = np.log1p(y_train)

def invert_log_safe(y_log):
    return np.expm1(np.clip(y_log, -20, 20))

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(train_df[categorical_features])

scaler_input = StandardScaler().fit(train_df[numeric_features].values)

def single_feature_mask_and_names(poly, input_names):
    all_names = list(poly.get_feature_names_out(input_names))
    mask = np.array([(" " not in n) and ("*" not in n) and (":" not in n) for n in all_names], dtype=bool)
    return mask, [n for n, m in zip(all_names, mask) if m]

Xtr_num_lin = scaler_input.transform(train_df[numeric_features].values)
Xte_num_lin = scaler_input.transform(test_df[numeric_features].values)
Xtr_cat = ohe.transform(train_df[categorical_features])
Xte_cat = ohe.transform(test_df[categorical_features])

Xtr_lin = np.hstack([Xtr_num_lin, Xtr_cat])
Xte_lin = np.hstack([Xte_num_lin, Xte_cat])

linear_model = LinearRegression()
linear_model.fit(Xtr_lin, y_train_log)

pred_train_log_lin = linear_model.predict(Xtr_lin)
pred_train_lin = invert_log_safe(pred_train_log_lin)
pred_train_lin = np.clip(pred_train_lin, 1e-6, None)
smear_lin = float(np.mean(y_train / pred_train_lin))

y_pred_lin_raw = linear_model.predict(Xte_lin)
y_pred_lin = invert_log_safe(y_pred_lin_raw)
y_pred_lin_corr = np.clip(y_pred_lin * smear_lin, 0, None)

lin_mse = mean_squared_error(y_test, y_pred_lin_corr)
lin_r2  = r2_score(y_test, y_pred_lin_corr)

cv = KFold(n_splits=5, shuffle=True, random_state=RND)

def eval_poly_no_interactions(degree, cv, clip_val=50.0, svd_components=None, alphas=None):
    Xtr_in = scaler_input.transform(train_df[numeric_features].values)
    Xte_in = scaler_input.transform(test_df[numeric_features].values)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr_all = poly.fit_transform(Xtr_in)
    Xte_all = poly.transform(Xte_in)

    mask, _ = single_feature_mask_and_names(poly, numeric_features)
    Xtr_single = np.clip(Xtr_all[:, mask], -clip_val, clip_val)
    Xte_single = np.clip(Xte_all[:, mask], -clip_val, clip_val)

    scaler_poly = StandardScaler().fit(Xtr_single)
    Xtr_single_s = scaler_poly.transform(Xtr_single)
    Xte_single_s = scaler_poly.transform(Xte_single)

    if svd_components and svd_components < Xtr_single_s.shape[1]:
        svd = TruncatedSVD(n_components=svd_components, random_state=RND)
        Xtr_reduced = svd.fit_transform(Xtr_single_s)
        Xte_reduced = svd.transform(Xte_single_s)
    else:
        Xtr_reduced = Xtr_single_s
        Xte_reduced = Xte_single_s

    Xtr_full = np.hstack([Xtr_reduced, ohe.transform(train_df[categorical_features])])
    Xte_full = np.hstack([Xte_reduced, ohe.transform(test_df[categorical_features])])

    if alphas is None:
        alphas = np.logspace(-6, 10, 40)

    ridge = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    ridge.fit(Xtr_full, y_train_log)

    pred_train = invert_log_safe(ridge.predict(Xtr_full))
    pred_train = np.clip(pred_train, 1e-6, None)
    smear = float(np.mean(y_train / pred_train))

    pred_test = invert_log_safe(ridge.predict(Xte_full))
    pred_test_corr = np.clip(pred_test * smear, 0, None)

    return {
        "degree": degree, "mse": mean_squared_error(y_test, pred_test_corr),
        "r2": r2_score(y_test, pred_test_corr), "alpha": float(ridge.alpha_),
        "n_num_feats": Xtr_reduced.shape[1], "interactions": False
    }

def eval_poly_with_interactions_deg2(cv, clip_val=50.0, svd_components=None, alphas=None):
    Xtr_in = scaler_input.transform(train_df[numeric_features].values)
    Xte_in = scaler_input.transform(test_df[numeric_features].values)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtr_all = np.clip(poly.fit_transform(Xtr_in), -clip_val, clip_val)
    Xte_all = np.clip(poly.transform(Xte_in), -clip_val, clip_val)

    scaler_poly = StandardScaler().fit(Xtr_all)
    Xtr_s = scaler_poly.transform(Xtr_all)
    Xte_s = scaler_poly.transform(Xte_all)

    if svd_components and svd_components < Xtr_s.shape[1]:
        svd = TruncatedSVD(n_components=svd_components, random_state=RND)
        Xtr_red = svd.fit_transform(Xtr_s)
        Xte_red = svd.transform(Xte_s)
    else:
        Xtr_red = Xtr_s
        Xte_red = Xte_s

    Xtr_full = np.hstack([Xtr_red, ohe.transform(train_df[categorical_features])])
    Xte_full = np.hstack([Xte_red, ohe.transform(test_df[categorical_features])])

    if alphas is None:
        alphas = np.logspace(-6, 8, 30)

    ridge = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    ridge.fit(Xtr_full, y_train_log)

    pred_train = invert_log_safe(ridge.predict(Xtr_full))
    pred_train = np.clip(pred_train, 1e-6, None)
    smear = float(np.mean(y_train / pred_train))

    pred_test = invert_log_safe(ridge.predict(Xte_full))
    pred_test_corr = np.clip(pred_test * smear, 0, None)

    return {
        "degree": 2, "mse": mean_squared_error(y_test, pred_test_corr),
        "r2": r2_score(y_test, pred_test_corr), "alpha": float(ridge.alpha_),
        "n_num_feats": Xtr_red.shape[1], "interactions": True
    }

res2 = eval_poly_no_interactions(2, cv)
res3 = eval_poly_no_interactions(3, cv, clip_val=30.0, svd_components=20)
res4 = eval_poly_no_interactions(4, cv, clip_val=15.0, svd_components=25)
res2_int = eval_poly_with_interactions_deg2(cv)

results = [
    ["Linear Regression", lin_mse, lin_r2, "N/A", "No"],
    ["Polynomail of Degree 2 (No Interactions)", res2["mse"], res2["r2"], res2["alpha"], "No"],
    ["Polynomail of Degree 2 (With Interactions)", res2_int["mse"], res2_int["r2"], res2_int["alpha"], "Yes"],
    ["Polynomail of Degree 3 (No Interactions)", res3["mse"], res3["r2"], res3["alpha"], "No"],
    ["Polynomail of Degree 4 (No Interactions)", res4["mse"], res4["r2"], res4["alpha"], "No"],
]
print("\nMODEL PERFORMANCE (TEST SET)\n")
print("{:<45} {:>12} {:>12} {:>16} {:>15}".format(
    "Model", "MSE", "R²", "Alpha", "Interactions"
))
print("-" * 110)

for row in results:
    model_name = row[0]
    mse_val = row[1]
    r2_val = row[2]

    if isinstance(row[3], str):
        alpha_value = row[3]
    else:
        alpha_value = f"{row[3]:.6f}"

    interaction_value = "Yes" if row[4] == "Yes" else "No"

    print("{:<45} {:>12.2f} {:>12.4f} {:>16} {:>15}".format(
        model_name, mse_val, r2_val, alpha_value, interaction_value
    ))
best_model = min(results, key=lambda x: x[1])  # lowest MSE

print("\nBEST MODEL INTERPRETATION BASED ON MSE VALUES\n")
print("-" * 50)

model_name = best_model[0]
mse_val = best_model[1]
r2_val = best_model[2]

if isinstance(best_model[3], str):
    alpha_value = best_model[3]
else:
    alpha_value = f"{best_model[3]:.6f}"

interaction_value = "Yes" if best_model[4] == "Yes" else "No"

print(f"Best Model       : {model_name}")
print(f"Test MSE         : {mse_val:.2f}")
print(f"Test R²          : {r2_val:.4f}")
print(f"Alpha            : {alpha_value}")
print(f"Interactions     : {interaction_value}")

print("-" * 50)